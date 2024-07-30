"""This module transforms the annotation projection/word alignment problems to
ILP problems based of word/token ranges and provide functions to solve them"""

from enum import Enum
from itertools import product
import logging
from typing import Callable, Tuple

import numpy as np
import cvxpy as cp
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from src.utils.entities import (
    get_overlapped_by_candidates,
    get_overlapped_candidates_idxs,
)

logger = logging.getLogger("Pipeline")

try:
    import gurobipy as grp
except ImportError:
    logger.warn("If you want to use GUROBI please install gurobipy")


def get_cosine_similarity_based_cost(
    src_emb: list[list[float]] | np.ndarray,
    tgt_emb: list[list[float]] | np.ndarray,
    src_spans: list[tuple[int, int]],
    tgt_spans: list[tuple[int, int]],
    threshold: float = 0.65,
    max_reduction: Callable[[np.ndarray], np.dtype] = np.mean,
) -> csr_matrix:
    """Compute matching costs between words based on cosine similarity between
    token-level embeddings. The calculation is performed in the following manner:
    for every pair of words it takes all token-level embeddings, for every source/
    target word token the maximum cosine similarity is computed among all target/source
    tokens. Them it reduce all maximum for word with a given reduction function.
    The cost of matching between words is computed by taking maximum of cost for target
    and source words and substracting a given threshold.

    Args:
        src_emb (list[list[float]] | np.ndarray): token-level source words embeddings
        tgt_emb (list[list[float]] | np.ndarray): token-level target words embeddings
        src_spans (list[tuple[int, int]]): token-level spans of source words
        tgt_spans (list[tuple[int, int]]): token-level spans of target words
        threshold (float, optional): minimum word-level similarity score to consider
            word can be aligned. Defaults to 0.8.
        max_reduction (Callable[[np.ndarray], np.dtype], optional): Operation to reduce
            maximums of cosine similarities among all tokens of the word.
            Defaults to np.mean.

    Returns:
        csr_matrix: matrix of costs between source words and target words
    """
    cos_dist = cdist(src_emb, tgt_emb, "cosine")
    cos_sim = 1 - cos_dist

    costs = np.empty((len(src_spans), len(tgt_spans)))
    for (i, src_span), (j, tgt_span) in product(
        enumerate(src_spans), enumerate(tgt_spans)
    ):
        sim_st = cos_sim[src_span[0] : src_span[1], tgt_span[0] : tgt_span[1]]

        cs = max_reduction(sim_st.max(axis=1))
        ct = max_reduction(sim_st.max(axis=0))
        cost = max(cs, ct) - threshold

        costs[i, j] = cost

    return csr_matrix(costs)


def get_relative_lenght_cost(
    alignments_by_src_words: list[list[int]],
    src_entity_spans: list[tuple[int, int]],
    tgt_candidates: list[tuple[int, int]],
) -> csr_matrix:
    """Compute matching costs between source entities and target candidates
    using word to word alignments. This implementation computes scrores as
    a number of aligned word pairs between a source entity and candidate
    divided by sum of number of words in a source entity and a candidate.
    The main idea is that if all entity/candidates words are aligned that
    it is probably a correct matching

    Args:
        alignments_by_src_words (list[list[int]]): list of list of word aligned to
            source words
        src_entity_spans (list[tuple[int, int]]): spans (in word indices) of
            source entities
        tgt_candidates (list[tuple[int, int]]): spans of candidates

    Returns:
        csr_matrix: matrix of costs between source entities and candidates.
            We use csr format because matrix is usually sparse.
    """

    weights = []
    col_indices = []
    row_ptr = [0]

    for src_s, src_e in src_entity_spans:
        src_aligns = alignments_by_src_words[src_s:src_e]
        src_ent_len = src_e - src_s
        for col, (tgt_s, tgt_e) in enumerate(tgt_candidates):
            num_aligned_words = 0
            for tgt_idxs in src_aligns:
                num_aligned_words += sum(
                    1 if tgt_s <= idx < tgt_e else 0 for idx in tgt_idxs
                )

            if num_aligned_words > 0:
                tgt_cand_len = tgt_e - tgt_s
                w = 2 * num_aligned_words / (src_ent_len + tgt_cand_len)

                weights.append(w)
                col_indices.append(col)

        row_ptr.append(len(col_indices))

    mat_shape = (len(src_entity_spans), len(tgt_candidates))
    return csr_matrix((weights, col_indices, row_ptr), shape=mat_shape)


class ProjectionContraint(Enum):
    EQUAL = "eq"
    LESS = "le"
    LESS_OR_EQUAL = "leq"
    GREATER = "gr"
    GREATER_OR_EQUAL = "geq"


def get_constraint_operation(proj_constraint: ProjectionContraint):
    match proj_constraint:
        case ProjectionContraint.EQUAL:
            return lambda a, b: a == b
        case ProjectionContraint.LESS:
            return lambda a, b: a < b
        case ProjectionContraint.LESS_OR_EQUAL:
            return lambda a, b: a <= b
        case ProjectionContraint.GREATER:
            return lambda a, b: a > b
        case ProjectionContraint.GREATER_OR_EQUAL:
            return lambda a, b: a >= b


def greedy_solve_from_costs(
    costs: csr_matrix,
    tgt_candidates: list[tuple[int, int]],
    n_projected: int = 1,
    proj_constraint: ProjectionContraint = ProjectionContraint.EQUAL,
) -> Tuple[list[int], list[int]]:
    n_ent, n_cand = costs.shape
    C = costs.toarray()

    num_projected_by_entity = [0 for _ in range(n_ent)]
    overlapped_by_cands = get_overlapped_by_candidates(tgt_candidates)

    ineq_constr = get_constraint_operation(proj_constraint)

    src_idxs, cand_idxs = [], []
    while C.max() > 0:
        # take item with maximum cost and add it to solution
        ent_idx, cand_idx = np.unravel_index(
            np.argmax(C.reshape(1, -1)), (n_ent, n_cand)
        )
        src_idxs.append(ent_idx.item())
        cand_idxs.append(cand_idx.item())

        # check whether constraints are fulfilled for entity
        num_projected_by_entity[ent_idx] += 1
        curr_n_proj = num_projected_by_entity[ent_idx]
        if proj_constraint in [
            ProjectionContraint.EQUAL,
            ProjectionContraint.LESS_OR_EQUAL,
        ]:
            # entity reach projection limit
            if ineq_constr(curr_n_proj, n_projected):
                C[ent_idx] = 0
        elif proj_constraint in [ProjectionContraint.LESS]:
            # next projection for this entity will violate constraints
            if not ineq_constr(curr_n_proj + 1, n_projected):
                C[ent_idx] = 0

        # remove overlapped candidates from possible solution
        overlapped = overlapped_by_cands[cand_idx]
        overlapped.append(cand_idx)
        C[:, overlapped] = 0

    return src_idxs, cand_idxs


def remove_entities_with_zero_cost(costs: csr_matrix) -> Tuple[csr_matrix, np.ndarray]:
    row_mask = costs.getnnz(1) > 0
    nnz_rows = np.argwhere(row_mask)[:, 0]
    costs = costs[costs.getnnz(1) > 0]

    return costs, nnz_rows


def remove_candidates_with_zero_costs(
    costs: csr_matrix,
) -> Tuple[csr_matrix, list[tuple[int, int]], np.ndarray]:
    col_mask = costs.getnnz(0) > 0
    nnz_cols = np.argwhere(col_mask)[:, 0]
    costs = costs[:, costs.getnnz(0) > 0]

    return costs, nnz_cols


def construct_ilp_problem(
    costs: csr_matrix,
    tgt_candidates: list[tuple[int, int]],
    n_projected: int = 1,
    proj_constraint: ProjectionContraint = ProjectionContraint.EQUAL,
) -> cp.Problem:
    """Construct ILP problem which can be solved by ILP solver with use of
    the solve_ilp_porblem function.

    Args:
        costs (csr_matrix): matrix of matching costst between source entities and
            target candidates
        tgt_candidates (list[tuple[int, int]]): list of target candidates in a form of
            spans. Used to add non-overlaping constraints
        n_projected (int, optional): Number of candidates to be projected from one
            source entities. Depends on proj_constraint param. Defaults to 1.
        proj_constraint (ProjectionContraint, optional): Determine type of constraint
            for number of projected entitites from one source entity. For example:
            default params mean that every source entitity should be projected to only
            one target candidate. Defaults to ProjectionContraint.EQUAL.
    Returns:
        cp.Problem: weighted bipartite matching problem with additional constratints
    """
    n_ent, n_cand = costs.shape
    n_total = n_ent * n_cand

    C = costs.toarray().reshape((1, -1))
    x = cp.Variable(C.shape[1], boolean=True)

    objective = cp.Maximize(C @ x)
    constraints = []

    # Constraint: all source entities should be projected
    const_op = get_constraint_operation(proj_constraint)
    for i in range(0, n_ent * n_cand, n_cand):
        n_proj_src_ent = sum(x[i : i + n_cand])
        constraints.append(const_op(n_proj_src_ent, n_projected))

    # Constraint: don't project enitities to overlapped candidates
    overlapped_cands = get_overlapped_candidates_idxs(tgt_candidates)
    # don't project to overlapped
    for a_idx, b_idx in overlapped_cands:
        n_proj_a = sum(x[a_idx:n_total:n_cand])
        n_proj_b = sum(x[b_idx:n_total:n_cand])
        constraints.append(n_proj_a + n_proj_b <= 1)

    problem = cp.Problem(objective=objective, constraints=constraints)
    return problem


def solve_ilp_problem(
    problem: cp.Problem,
    n_ent: int,
    n_cand: int,
    solver: str = cp.GUROBI,
    **solver_params,
) -> Tuple[list[int], list[int]]:
    """Solve weighted bipartite matching problem and return indices of mathed source
    entities and target candidates

    Args:
        problem (cp.Problem): ILP problem generated by construct_ilp_problem
        n_ent (int): number of source entities in the problem
        n_cand (np.ndarray): number of target candidates in the problem
        solver (str, optional): solver's name to use. Defaults to cp.GUROBI.

    Returns:
        tuple[list[int], list[int]]: the first item of a tuple contains indices of
            matched source entities whereas the second - indices of matched
            target candidates
    """
    problem.solve(solver=solver, **solver_params)

    x = problem.variables()[0].value
    if x is None:
        logger.warn(f"ILP solver failed. Status: {problem.status}")
        return [], []

    idxs = np.argwhere(x == 1)[:, 0]
    src_idxs, cand_idxs = np.unravel_index(idxs, (n_ent, n_cand))

    return src_idxs.tolist(), cand_idxs.tolist()


def solve_ilp_with_gurobi(
    costs: csr_matrix,
    tgt_candidates: list[tuple[int, int]],
    n_projected: int = 1,
    proj_constraint: ProjectionContraint = ProjectionContraint.EQUAL,
    **solver_params,
) -> Tuple[list[int], list[int]]:
    """Construct and solve range based ILP with use of GUROBI. In general CVXPY supports
    GUROBI as a solver, but before calling GUROBI itself it performs some recuductions
    that can freeze/consume a lot of memory for some problems. The purpose of this
    function is to mitigate this issues and call GUROBI directly.

    Args:
        costs (csr_matrix): matrix of matching costst between source entities and
            target candidates
        tgt_candidates (list[tuple[int, int]]): list of target candidates in a form of
            spans. Used to add non-overlaping constraints
        n_projected (int, optional): Number of candidates to be projected from one
            source entities. Depends on proj_constraint param. Defaults to 1.
        proj_constraint (ProjectionContraint, optional): Determine type of constraint
            for number of projected entitites from one source entity. For example:
            default params mean that every source entitity should be projected to only
            one target candidate. Defaults to ProjectionContraint.EQUAL.

    Returns:
        Tuple[list[int], list[int]]: the first item of a tuple contains indices of
            matched source entities whereas the second - indices of matched
            target candidates. The same format as for other solvers
    """

    n_ent, n_cand = costs.shape
    C = costs.toarray()

    proj_const_op = get_constraint_operation(proj_constraint)

    overlapped_cands = get_overlapped_candidates_idxs(tgt_candidates)

    src_idxs, cand_idxs = [], []
    with grp.Model("nlp_ilp") as m:
        m.setParam("LogToConsole", 0)
        for key, value in solver_params.items():
            m.setParam(key, value)

        # create variables
        x = m.addVars(n_ent, n_cand, vtype=grp.GRB.BINARY, name="projection")

        # add objective function
        m.setObjective(
            grp.quicksum(x[i, j] * C[i, j] for i, j in np.ndindex(n_ent, n_cand)),
            grp.GRB.MAXIMIZE,
        )

        # constraint on number of times every entity should be projected
        m.addConstrs(
            (proj_const_op(x.sum(i, "*"), n_projected) for i in range(n_ent)),
            name="num_projection",
        )

        # don't project to overlapped candidates
        for a_idx, b_idx in overlapped_cands:
            m.addConstr(
                x.sum("*", a_idx) + x.sum("*", b_idx) <= 1,
                name="not_project_to_overlapped",
            )

        m.optimize()

        if m.Status in [
            grp.GRB.OPTIMAL,
            grp.GRB.SUBOPTIMAL,
            grp.GRB.ITERATION_LIMIT,
            grp.GRB.NODE_LIMIT,
            grp.GRB.TIME_LIMIT,
            grp.GRB.SOLUTION_LIMIT,
            grp.GRB.USER_OBJ_LIMIT,
            grp.GRB.WORK_LIMIT,
            grp.GRB.MEM_LIMIT,
            grp.GRB.NUMERIC,
        ]:
            for i, j in np.ndindex(n_ent, n_cand):
                if x[i, j].X > 0:
                    src_idxs.append(i)
                    cand_idxs.append(j)

    return src_idxs, cand_idxs
