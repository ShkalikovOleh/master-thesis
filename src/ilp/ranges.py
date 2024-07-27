"""This module transforms the annotation projection/word alignment problem to
an ILP problem based of word/token ranges and give function to solve it"""

from enum import Enum
from itertools import combinations, combinations_with_replacement
import logging
from typing import Tuple

import numpy as np
import cvxpy as cp
from scipy.sparse import csr_matrix

from src.utils.entities import (
    get_overlapped_by_candidates,
    get_overlapped_candidates_idxs,
)

logger = logging.getLogger("Pipeline")


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
    it is probably a coorrect matching

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


def construct_ilp_problem(
    costs: csr_matrix,
    tgt_candidates: list[tuple[int, int]],
    n_projected: int = 1,
    proj_constraint: ProjectionContraint = ProjectionContraint.EQUAL,
    remove_entities_with_zero_cost: bool = True,
    remove_candidates_with_zero_cost: bool = True,
) -> Tuple[cp.Problem, np.ndarray, np.ndarray]:
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
        remove_entities_with_zero_cost (bool, optional): whether consider or not
            entities that don't have any non-zero cost
        remove_candidates_with_zero_cost (bool, optional): whether consider or not
            candidates that don't have any non-zero cost
    Returns:
        Tuple[cp.Problem, np.ndarray, np.ndarray]: tuple of weighted bipartite matching
            problem with additional constratints, indices of source entities and
            candidates that will be considerd by the problem

    """
    if remove_entities_with_zero_cost:
        row_mask = costs.getnnz(1) > 0
        nnz_rows = np.argwhere(row_mask)[:, 0]
        costs = costs[costs.getnnz(1) > 0]
    else:
        nnz_rows = np.arange(costs.shape[0])

    if remove_candidates_with_zero_cost:
        col_mask = costs.getnnz(0) > 0
        nnz_cols = np.argwhere(col_mask)[:, 0]
        costs = costs[:, costs.getnnz(0) > 0]

        tgt_candidates = list(map(tgt_candidates.__getitem__, nnz_cols))
    else:
        nnz_cols = np.arange(costs.shape[1])

    n_ent, n_cand = costs.shape

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
    # don't project different entities to the same candidate
    for src_ent1, src_ent2 in combinations(range(n_ent), 2):
        off1 = src_ent1 * n_cand
        off2 = src_ent2 * n_cand
        for idx in range(n_cand):
            constraints.append(x[off1 + idx] + x[off2 + idx] <= 1)
    # don't project to overlapped
    for src_ent1, src_ent2 in combinations_with_replacement(range(n_ent), 2):
        off1 = src_ent1 * n_cand
        off2 = src_ent2 * n_cand
        # do it for every entity pair
        for a_idx, b_idx in overlapped_cands:
            constraints.append(x[off1 + a_idx] + x[off2 + b_idx] <= 1)
            constraints.append(x[off1 + b_idx] + x[off2 + a_idx] <= 1)

    problem = cp.Problem(objective=objective, constraints=constraints)
    return problem, nnz_rows, nnz_cols


def solve_ilp_problem(
    problem: cp.Problem,
    nnz_rows: np.ndarray,
    nnz_cols: np.ndarray,
    solver: str = cp.GUROBI,
    **solver_params,
) -> Tuple[list[int], list[int]]:
    """Solve weighted bipartite matching problem and return indices of mathed source
    entities and target candidates

    Args:
        problem (cp.Problem): ILP problem generated by construct_ilp_problem
        nnz_rows (np.ndarray): indices of non zero rows of cost matrix obtained from
            construct_ilp_problem
        nnz_cols (np.ndarray): indices of non zero columns of cost matrix obtained from
            construct_ilp_problem
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
    src_idxs, cand_idxs = np.unravel_index(idxs, (len(nnz_rows), len(nnz_cols)))

    return nnz_rows[src_idxs].tolist(), nnz_cols[cand_idxs].tolist()
