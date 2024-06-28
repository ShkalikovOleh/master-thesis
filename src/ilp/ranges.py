"""This module transforms the annotation projection/word alignment problem to
an ILP problem based of word/token ranges and give function to solve it"""

from enum import Enum

import numpy as np
import cvxpy as cp
from scipy.sparse import csr_matrix

from src.utils.entities import get_overlapped_candidates_idxs


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
    for i in range(0, n_ent * n_cand, n_cand):
        # do it for every entity
        for a_idx, b_idx in overlapped_cands:
            constraints.append(x[i + a_idx] + x[i + b_idx] <= 1)

    problem = cp.Problem(objective=objective, constraints=constraints)
    return problem


def solve_ilp_problem(
    problem: cp.Problem, n_src_entities: int, n_tgt_cands: int, solver: str = cp.GUROBI
) -> tuple[list[int], list[int]]:
    """Solve weighted bipartite matching problem and return indices of mathed source
    entities and target candidates

    Args:
        problem (cp.Problem): ILP problem generated by construct_ilp_problem
        n_src_entities (int): number of source entities
        n_tgt_cands (int): number of target candidates
        solver (str, optional): solver's name to use. Defaults to cp.GUROBI.

    Returns:
        tuple[list[int], list[int]]: the first item of a tuple contains indices of
            matched source entities whereas the second - indices of matched
            target candidates
    """
    problem.solve(solver=solver)
    x = problem.variables()[0].value

    idxs = np.argwhere(x == 1)[:, 0]
    src_idxs, cand_idxs = np.unravel_index(idxs, (n_src_entities, n_tgt_cands))

    return src_idxs.tolist(), cand_idxs.tolist()
