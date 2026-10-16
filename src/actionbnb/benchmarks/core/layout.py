"""Decision-vector layout of a horizon MIQP.

Over an N-stage horizon, the decision vector stacks four groups:

    z = [x_1, ..., x_N,  u_0, ..., u_{N-1},  delta_0, ..., delta_{N-1},  aux_0, ..., aux_{N-1}]

Stage i = 0, ..., N-1 owns the predicted state x_{i+1}, the input u_i, the
binaries delta_i, and continuous auxiliaries aux_i (e.g. epigraph variables).
The measured state x_0 is a parameter, not a decision variable.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse as sp

GROUPS = ("x", "u", "delta", "aux")


def stages(block, N: int) -> sp.csr_array:
    """Repeat one stage block N times along the diagonal, or place N given stage blocks (N, rows, cols)."""
    if np.ndim(block) == 3:
        return sp.block_diag([sp.csr_array(b) for b in block], format="csr")
    return sp.kron(sp.eye_array(N), sp.csr_array(block), format="csr")


def shifted(block, N: int) -> sp.csr_array:
    """Place one stage block below the diagonal: row block i uses column block i - 1."""
    return sp.kron(sp.eye_array(N, k=-1), sp.csr_array(block), format="csr")


@dataclass(frozen=True)
class StageLayout:
    """Column layout of z = [X, U, Delta, Aux] over an N-stage horizon."""

    N: int
    n_x: int
    n_u: int
    n_delta: int = 0
    n_aux: int = 0

    def size(self, group: str) -> int:
        """Per-stage size of a group."""
        return (self.n_x, self.n_u, self.n_delta, self.n_aux)[GROUPS.index(group)]

    def width(self, group: str) -> int:
        """Number of columns of a group over the horizon."""
        return self.N * self.size(group)

    @property
    def n_z(self) -> int:
        """Number of decision variables."""
        return sum(self.width(g) for g in GROUPS)

    def index(self, group: str, i: int | None = None) -> np.ndarray:
        """Columns of a group, or of its stage-i block (for x, that is x_{i+1})."""
        start = sum(self.width(g) for g in GROUPS[:GROUPS.index(group)])
        if i is None:
            return np.arange(start, start + self.width(group))
        size = self.size(group)
        return np.arange(start + i * size, start + (i + 1) * size)

    @property
    def binary_idx(self) -> np.ndarray:
        return self.index("delta")

    def vector(self, x=None, u=None, delta=None, aux=None) -> np.ndarray:
        """Stack group values into z; each is a stage vector (repeated), all N stages, or zero if omitted."""
        parts = []
        for group, part in zip(GROUPS, (x, u, delta, aux)):
            shape = (self.N, self.size(group))
            parts.append(np.zeros(shape) if part is None else np.broadcast_to(np.asarray(part, float), shape))
        return np.concatenate([part.ravel() for part in parts])

    def columns(self, x=None, u=None, delta=None, aux=None) -> sp.csr_array:
        """Place group column blocks side by side; omitted groups are zero."""
        blocks = (x, u, delta, aux)
        n_rows = next(block.shape[0] for block in blocks if block is not None)
        return sp.hstack(
            [
                sp.csr_array((n_rows, self.width(g))) if block is None else sp.csr_array(block)
                for g, block in zip(GROUPS, blocks)
                if self.width(g) > 0
            ],
            format="csr",
        )
