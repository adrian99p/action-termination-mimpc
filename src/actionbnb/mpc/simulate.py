"""Nominal closed-loop simulation of a benchmark under a controller."""

from dataclasses import dataclass, field

import numpy as np

from actionbnb.benchmarks import Benchmark
from actionbnb.mpc.controllers import Controller, Decision


@dataclass(slots=True)
class ClosedLoop:
    """States x_0, ..., x_T and the decisions at instants 0, ..., T-1."""

    states: list[np.ndarray] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)

    @property
    def inputs(self) -> np.ndarray:
        return np.array([d.u for d in self.decisions])

    @property
    def runtimes(self) -> np.ndarray:
        return np.array([d.runtime for d in self.decisions])


def simulate(benchmark: Benchmark, controller: Controller, steps: int) -> ClosedLoop:
    """Apply the controller's first input to the nominal plant at each of `steps` sampling instants."""
    if not benchmark.closed_loop:
        raise ValueError(f"{benchmark.name} is defined over one horizon only.")
    x, u_prev = benchmark.initial_state(), benchmark.initial_input()
    loop = ClosedLoop(states=[x])
    for k in range(steps):
        decision = controller(benchmark.problem(x, k, u_prev))
        if decision.u is None:
            raise RuntimeError(f"No input at instant {k}: {decision.status}.")
        x, u_prev = benchmark.step(x, decision.u, k), np.asarray(decision.u)
        loop.states.append(x)
        loop.decisions.append(decision)
    return loop
