"""This module transforms the annotation projection/word alignment problems to
ILP problems based of word/token ranges and provide functions to solve them"""

from enum import Enum
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

try:
    import gurobipy as grp
except ImportError:
    logger.warn("If you want to use GUROBI please install gurobipy")


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
    total_cost = 0
    while C.max() > 0:
        # take item with maximum cost and add it to solution
        ent_idx, cand_idx = np.unravel_index(
            np.argmax(C.reshape(1, -1)), (n_ent, n_cand)
        )
        total_cost += C[ent_idx, cand_idx]
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

    return src_idxs, cand_idxs, total_cost


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
        return [], [], 0

    idxs = np.argwhere(x == 1)[:, 0]
    src_idxs, cand_idxs = np.unravel_index(idxs, (n_ent, n_cand))

    return src_idxs.tolist(), cand_idxs.tolist(), problem.objective.value


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
            total_cost = m.getObjective().getValue()
        else:
            total_cost = 0

    return src_idxs, cand_idxs, total_cost
