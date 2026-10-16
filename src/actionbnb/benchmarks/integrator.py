"""Double integrator with three-level acceleration.

With position p and velocity v, x = [p, v] and the MPC problem is

    min  sum_{i=0}^{N-1} x_{i+1}' Q x_{i+1} + R u_i^2
    s.t. x_{i+1} = A x_i + B u_i   (exact zero-order hold, DT = 0.5),
         u_i in {-1, 0, 1}.

There are no state constraints or terminal terms. The seed samples a braking
transient: position and velocity share a random sign, with magnitudes in
[2, 4] and [0.5, 1.5]. A constructed example, not a literature benchmark.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from actionbnb.benchmarks.core import Benchmark, HorizonBuilder, StageLayout, three_level
from actionbnb.problem import MIQPProblem

DT = 0.5
A = np.array([[1.0, DT], [0.0, 1.0]])
B = np.array([[0.5 * DT**2], [DT]])
Q = np.diag([1.0, 0.1])
R = np.array([[0.01]])
INPUTS = three_level()

# Magnitude ranges of the initial position and velocity.
MAGNITUDE_MIN = np.array([2.0, 0.5])
MAGNITUDE_MAX = np.array([4.0, 1.5])


@dataclass(frozen=True)
class Integrator(Benchmark):
    name: ClassVar[str] = "integrator"
    seed: int = 0
    N: int = 20

    @property
    def layout(self) -> StageLayout:
        return StageLayout(self.N, n_x=2, n_u=1, n_delta=INPUTS.n_delta)

    def dynamics(self, k: int = 0) -> tuple[np.ndarray, np.ndarray]:
        return A, B

    def initial_state(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        magnitude = rng.uniform(MAGNITUDE_MIN, MAGNITUDE_MAX)
        return rng.choice([-1.0, 1.0]) * magnitude

    def problem(self, x: np.ndarray, k: int = 0, u_prev: np.ndarray | None = None) -> MIQPProblem:
        """Time-invariant and memoryless: k only labels the problem, and u_prev is unused."""
        builder = HorizonBuilder(self.layout)
        builder.dynamics(A, B, x)
        builder.discrete_input(INPUTS)
        builder.tracking_cost(Q)
        builder.input_cost(R)
        return builder.build(f"{self.name}_seed{self.seed}_N{self.N}_k{k}")
