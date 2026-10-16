"""Action-certification diagnostic: an easy first input with a hard tail.

A scalar integrator with binary inputs, written in the deviation e = x - x_r from
the reference to avoid cancellation:

    min  Q_ACTION (u_0 - THETA)^2 + Q_TERMINAL e_N^2
    s.t. e_{i+1} = e_i + w_i u_i,   e_0 = -x_r,   u_i in {0, 1}.

With w_0 = 1 and distinct integer tail weights, the tail is a subset-sum problem.
The reference x_r = 1 + sum_{odd i} w_i + 1/2 lies half a unit above the state
reached by u_0 = 1 and the odd tail inputs, so no integer solution reaches it.
For every seed and N, the root relaxation has cost 0, fixing u_0 = 0 or u_0 = 1
gives bounds 810 or 10, and the optimum is 10.25 with u_0 = 1. Any incumbent
below 810 therefore certifies u_0 = 1, however poor its tail.

A constructed diagnostic, not a literature benchmark, and defined over one
horizon only (open loop). Seed 0 uses the stored weights; other seeds draw
N - 1 distinct integer tail weights in [TAIL_MIN, TAIL_MAX].
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from actionbnb.benchmarks.core import Benchmark, HorizonBuilder, StageLayout, on_off
from actionbnb.problem import MIQPProblem

Q_ACTION = 1000.0
THETA = 0.9
Q_TERMINAL = 1.0
TAIL_MIN, TAIL_MAX = 2500, 4999
INPUTS = on_off()

WEIGHTS = np.array([
    1.0, 2723.0, 4434.0, 4136.0, 3597.0, 3582.0, 4646.0, 2714.0,
    4243.0, 3003.0, 2735.0, 3816.0, 4939.0, 4339.0, 4402.0, 4293.0,
    4465.0, 3783.0, 2820.0, 4599.0, 3625.0, 3750.0, 3426.0, 2956.0,
    4816.0, 4453.0, 4109.0, 3506.0, 4556.0, 3863.0, 3608.0, 3626.0,
    3068.0, 2730.0, 3886.0, 4719.0, 2659.0, 4645.0, 4569.0, 3191.0,
])


@dataclass(frozen=True)
class Diagnostic(Benchmark):
    name: ClassVar[str] = "diagnostic"
    closed_loop: ClassVar[bool] = False
    seed: int = 0
    N: int = 16

    @property
    def layout(self) -> StageLayout:
        return StageLayout(self.N, n_x=1, n_u=1, n_delta=INPUTS.n_delta)

    @property
    def weights(self) -> np.ndarray:
        """w_0, ..., w_{N-1}, with w_0 = 1."""
        if self.seed == 0:
            return WEIGHTS[:self.N].copy()
        rng = np.random.default_rng(self.seed)
        tail = rng.choice(np.arange(TAIL_MIN, TAIL_MAX + 1), size=self.N - 1, replace=False)
        return np.r_[1.0, tail.astype(float)]

    def dynamics(self, k: int = 0) -> tuple[np.ndarray, np.ndarray]:
        return np.ones((1, 1)), self.weights[k].reshape(1, 1)

    def initial_state(self) -> np.ndarray:
        w = self.weights
        return np.array([-(w[0] + w[1::2].sum() + 0.5)])

    def problem(self, x: np.ndarray, k: int = 0, u_prev: np.ndarray | None = None) -> MIQPProblem:
        """Defined at k = 0 only; u_prev is unused."""
        N = self.N
        R = np.zeros((N, 1, 1))
        R[0] = Q_ACTION

        builder = HorizonBuilder(self.layout)
        builder.dynamics(np.ones((1, 1)), self.weights.reshape(N, 1, 1), x)
        builder.discrete_input(INPUTS)
        builder.input_cost(R, ref=THETA)
        builder.terminal_cost([[Q_TERMINAL]])
        return builder.build(f"{self.name}_seed{self.seed}_N{N}_k{k}")
