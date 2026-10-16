"""Convex mixed-integer quadratic problem representation."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_array

from actionbnb.action_encoding import ActionEncoding


@dataclass(slots=True)
class MIQPProblem:
    """Data for a convex MIQP with binary decisions: the MPC problem at one sampling instant.

    Minimize:
        0.5 * z.T @ P @ z + q.T @ z + constant

    Subject to:
        A @ z == b
        G @ z <= h
        lb <= z <= ub
        z[binary_idx] belongs to {0, 1}
    """

    P: csr_array
    q: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    binary_idx: np.ndarray

    A: csr_array | None = None
    b: np.ndarray | None = None
    G: csr_array | None = None
    h: np.ndarray | None = None

    constant: float = 0.0
    name: str = "miqp"
    action_encoding: ActionEncoding | None = None

    @property
    def n(self) -> int:
        """Number of decision variables."""
        return self.q.size

    @property
    def n_binary(self) -> int:
        """Number of binary variables."""
        return self.binary_idx.size

    