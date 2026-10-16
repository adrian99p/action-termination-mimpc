"""Results from the Gurobi oracle."""

from dataclasses import dataclass

import numpy as np

from actionbnb.status import SolveStatus


@dataclass
class OracleResult:
    """Result of a Gurobi solve used as ground truth."""

    status: SolveStatus
    objective: float | None
    z: np.ndarray | None
    runtime: float
    nodes_explored: int
    n_simplex: int
