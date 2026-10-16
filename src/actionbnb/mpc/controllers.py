"""Controllers: functions that return the first input to apply for a horizon MIQP."""

from collections.abc import Callable
from dataclasses import dataclass

from actionbnb.bnb import BnBConfig, solve
from actionbnb.oracle import solve_with_oracle
from actionbnb.problem import MIQPProblem
from actionbnb.status import SolveStatus


@dataclass(slots=True)
class Decision:
    """Applied input of one sampling instant and the solve that produced it."""

    u: tuple[float, ...] | None
    status: SolveStatus
    runtime: float
    nodes: int


Controller = Callable[[MIQPProblem], Decision]


def gurobi(time_limit: float = 60.0) -> Controller:
    """Full MIQP solve by Gurobi."""

    def control(problem: MIQPProblem) -> Decision:
        r = solve_with_oracle(problem, time_limit=time_limit)
        u = None if r.z is None else problem.action_encoding.action(r.z)
        return Decision(u, r.status, r.runtime, r.nodes_explored)

    return control


def bnb(config: BnBConfig) -> Controller:
    """Branch and bound with the given configuration, e.g. with action termination."""

    def control(problem: MIQPProblem) -> Decision:
        r = solve(problem, config)
        return Decision(r.action, r.status, r.runtime, r.nodes)

    return control
