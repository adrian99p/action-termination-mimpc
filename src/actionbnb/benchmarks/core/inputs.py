"""Finite input sets and their binary encodings.

A discrete input maps the binaries delta_i of one stage to the complete input
u_i = S delta_i. The admissible binary patterns are those satisfying the stage
rows G delta_i <= h. They are enumerated, so the action encoding contains every
pattern a feasible solution can take.
"""

from dataclasses import dataclass
from functools import cached_property
from itertools import product

import numpy as np

from actionbnb.action_encoding import ActionEncoding
from actionbnb.benchmarks.core.layout import StageLayout

MAX_STAGE_BINARIES = 20  # Patterns are enumerated, so 2**n_delta must stay small.


@dataclass(frozen=True, eq=False)
class DiscreteInput:
    """Finite input set {S delta : delta in {0, 1}^n_delta, G delta <= h}."""

    S: np.ndarray  # (n_u, n_delta)
    G: np.ndarray  # (n_rows, n_delta)
    h: np.ndarray  # (n_rows,)

    @property
    def n_u(self) -> int:
        return self.S.shape[0]

    @property
    def n_delta(self) -> int:
        return self.S.shape[1]

    @cached_property
    def patterns(self) -> tuple[tuple[int, ...], ...]:
        """Admissible binary patterns of one stage."""
        if self.n_delta > MAX_STAGE_BINARIES:
            raise ValueError(f"Cannot enumerate 2**{self.n_delta} binary patterns.")
        candidates = np.array(list(product((0, 1), repeat=self.n_delta)), dtype=float)
        admissible = np.all(candidates @ self.G.T <= self.h, axis=1)
        return tuple(tuple(int(v) for v in p) for p in candidates[admissible])

    def value(self, pattern) -> tuple[float, ...]:
        """Input encoded by a binary pattern."""
        return tuple(float(v) + 0.0 for v in self.S @ np.asarray(pattern, dtype=float))  # + 0.0 drops -0.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Componentwise input bounds over the admissible patterns."""
        values = np.array([self.value(p) for p in self.patterns])
        return values.min(axis=0), values.max(axis=0)

    def encoding(self, layout: StageLayout) -> ActionEncoding:
        """Encoding of the first input u_0 by the stage-0 binaries."""
        return ActionEncoding(
            indices=tuple(layout.index("delta", 0).tolist()),
            patterns={p: self.value(p) for p in self.patterns},
        )


def three_level() -> DiscreteInput:
    """Scalar u = delta+ - delta- in {-1, 0, 1}, with delta+ + delta- <= 1 and binaries (delta+, delta-)."""
    return DiscreteInput(S=np.array([[1.0, -1.0]]), G=np.array([[1.0, 1.0]]), h=np.ones(1))


def on_off(n_u: int = 1) -> DiscreteInput:
    """Each component u_j = delta_j in {0, 1}, e.g. pumps, valves, or thrusters."""
    return DiscreteInput(S=np.eye(n_u), G=np.zeros((0, n_u)), h=np.zeros(0))
