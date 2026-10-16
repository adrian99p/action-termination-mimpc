"""Translation from Gurobi status codes to SolveStatus."""

from gurobipy import GRB

from actionbnb.status import SolveStatus


_STATUS_MAP: dict[int, SolveStatus] = {
    GRB.OPTIMAL: SolveStatus.OPTIMAL,
    GRB.INFEASIBLE: SolveStatus.INFEASIBLE,
    GRB.UNBOUNDED: SolveStatus.UNBOUNDED,
    GRB.INF_OR_UNBD: SolveStatus.INFEASIBLE_OR_UNBOUNDED,
    GRB.CUTOFF: SolveStatus.CUTOFF,
    GRB.SUBOPTIMAL: SolveStatus.SUBOPTIMAL,
    GRB.TIME_LIMIT: SolveStatus.TIME_LIMIT,
    GRB.NODE_LIMIT: SolveStatus.NODE_LIMIT,
    GRB.ITERATION_LIMIT: SolveStatus.ITERATION_LIMIT,
    GRB.SOLUTION_LIMIT: SolveStatus.SOLUTION_LIMIT,
    GRB.USER_OBJ_LIMIT: SolveStatus.OBJECTIVE_LIMIT,
    GRB.WORK_LIMIT: SolveStatus.WORK_LIMIT,
    GRB.MEM_LIMIT: SolveStatus.MEMORY_LIMIT,
    GRB.INTERRUPTED: SolveStatus.INTERRUPTED,
    GRB.NUMERIC: SolveStatus.NUMERICAL_FAILURE,
}


def status_from_gurobi(status: int) -> SolveStatus:
    """Convert a Gurobi status to a SolveStatus."""
    return _STATUS_MAP.get(status, SolveStatus.SOLVER_FAILURE)
