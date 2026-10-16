"""Results of a branch-and-bound run."""

from dataclasses import dataclass, field
from math import inf

import numpy as np

from actionbnb.status import SolveStatus


@dataclass(slots=True)
class BnBTrace:
    """Bounds U, L, and L_{!=a} before each processed node and once more at termination."""

    nodes: list[int] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    competing: list[float] = field(default_factory=list)


@dataclass(slots=True)
class BnBResult:
    """Incumbent, final bounds, and search statistics."""

    status: SolveStatus
    z: np.ndarray | None = None
    objective: float = inf
    action: tuple[float, ...] | None = None
    lower_bound: float = -inf
    competing_bound: float | None = None

    runtime: float = 0.0
    nodes: int = 0
    qp_solves: int = 0
    qp_iterations: int = 0
    qp_time: float = 0.0

    first_incumbent_node: int | None = None
    first_incumbent_time: float | None = None
    certified_node: int | None = None
    certified_time: float | None = None

    trace: BnBTrace | None = field(default=None, repr=False)
