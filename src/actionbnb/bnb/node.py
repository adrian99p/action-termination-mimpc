"""Branch-and-bound tree nodes and the QP solutions they carry."""

from dataclasses import dataclass, field
from math import inf

import numpy as np

FREE = -1

Basis = tuple[np.ndarray, np.ndarray, int, int]


@dataclass(slots=True)
class QPSolution:
    """Optimal solution of a node QP."""

    objective: float
    z: np.ndarray
    basis: Basis | None = field(default=None, repr=False)


@dataclass(slots=True, eq=False)
class Node:
    """Subproblem defined by binary fixings, with a lower bound on its cost."""

    fixed: np.ndarray
    bound: float = -inf
    basis: Basis | None = field(default=None, repr=False)
    mask: int | None = None
    solution: QPSolution | None = field(default=None, repr=False)
    branch_var: int = -1
    branch_dist: float = 0.0
    is_open: bool = True

    @classmethod
    def root(cls, n_binary: int, mask: int | None = None) -> "Node":
        return cls(fixed=np.full(n_binary, FREE, dtype=np.int8), mask=mask)

    def child(self, j: int, value: int, bound: float, basis: Basis | None, dist: float, mask: int | None = None) -> "Node":
        fixed = self.fixed.copy()
        fixed[j] = value
        return Node(fixed, bound, basis, mask, branch_var=j, branch_dist=dist)
