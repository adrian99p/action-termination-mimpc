"""Common interface of the MI-MPC benchmarks."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from scipy.linalg import expm

from actionbnb.benchmarks.core.layout import StageLayout
from actionbnb.problem import MIQPProblem


def zero_order_hold(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact discretization of dx/dt = A_c x + B_c u with piecewise-constant u."""
    n, m = B_c.shape
    M = expm(np.block([[A_c, B_c], [np.zeros((m, n + m))]]) * dt)
    return M[:n, :n], M[:n, n:]


class Benchmark(ABC):
    """An MI-MPC benchmark: a nominal linear plant and its horizon MIQP.

    The seed fixes the instance data, such as the initial state or a reference
    phase. A closed loop starts from x = initial_state() and u_prev =
    initial_input(), and at each sampling instant k = 0, 1, ...

        problem = benchmark.problem(x, k, u_prev)
        u = problem.action_encoding.action(z)    # z solves problem
        x, u_prev = benchmark.step(x, u, k), u

    The open-loop instance is the problem at k = 0 (see instance()). Benchmarks
    defined only over one horizon set closed_loop = False.

    Subclasses are frozen dataclasses with at least the fields seed and N.
    """

    name: ClassVar[str]
    closed_loop: ClassVar[bool] = True
    seed: int
    N: int

    @property
    @abstractmethod
    def layout(self) -> StageLayout:
        """Column layout of the decision vector."""

    @abstractmethod
    def dynamics(self, k: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Matrices (A, B) of the nominal plant x_{k+1} = A x_k + B u_k at instant k."""

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Plant state at k = 0."""

    def initial_input(self) -> np.ndarray:
        """Input u_{-1} applied before k = 0; zero unless overridden."""
        return np.zeros(self.layout.n_u)

    @abstractmethod
    def problem(self, x: np.ndarray, k: int = 0, u_prev: np.ndarray | None = None) -> MIQPProblem:
        """Horizon MIQP at sampling instant k for the measured state x and previous input u_prev.

        u_prev defaults to initial_input(). Benchmarks without time-varying data or
        input memory ignore k or u_prev.
        """

    def instance(self) -> MIQPProblem:
        """Open-loop instance: the problem at k = 0."""
        return self.problem(self.initial_state(), 0, self.initial_input())

    def step(self, x: np.ndarray, u, k: int = 0) -> np.ndarray:
        """Nominal plant state after applying input u at state x and instant k."""
        A, B = self.dynamics(k)
        return A @ np.asarray(x, dtype=float) + B @ np.atleast_1d(np.asarray(u, dtype=float))
