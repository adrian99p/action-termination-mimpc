"""Shared outcomes for the oracle, node relaxations, and B&B search."""

from enum import StrEnum


class SolveStatus(StrEnum):
    """Outcome of an optimization solve."""

    # Successful / mathematical outcomes
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    CUTOFF = "cutoff"
    SUBOPTIMAL = "suboptimal"

    # Search termination
    ACTION_SATISFIED = "action_satisfied"
    TIME_LIMIT = "time_limit"
    NODE_LIMIT = "node_limit"
    ITERATION_LIMIT = "iteration_limit"
    SOLUTION_LIMIT = "solution_limit"
    OBJECTIVE_LIMIT = "objective_limit"
    WORK_LIMIT = "work_limit"
    MEMORY_LIMIT = "memory_limit"
    INTERRUPTED = "interrupted"

    # Failures
    NUMERICAL_FAILURE = "numerical_failure"
    SOLVER_FAILURE = "solver_failure"
