"""Settings of one branch-and-bound run."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class BnBConfig:
    """Search strategy, termination, tolerances, and limits."""

    branching: Literal["FF", "MF", "PC"] = "MF"  # First-fractional, most-fractional, or pseudocost
    selection: Literal["BFS", "DPLG"] = "BFS"  # Best-bound selection or deep plunging
    child_order: Literal["up", "nearest"] = "up"  # Preferred child: up branch or nearest the relaxation (N)
    rho: float = 0.25  # Plunging parameter of DPLG
    action_first: bool = False  # Action-first branching (AF)
    action_termination: bool = False  # Action termination (AT)

    eps: float = 0.0  # Termination tolerance epsilon
    int_tol: float = 1e-6
    bound_tol: float = 1e-8  # Prune and stop at U - tau with tau = eps + bound_tol

    time_limit: float | None = None  # s
    trace: bool = False  # Record U, L, and L_{!=a} before every node

    @property
    def name(self) -> str:
        """Label as in the paper, e.g. 'MF-DPLG-N-AF-AT'."""
        parts = [self.branching, self.selection]
        if self.child_order == "nearest":
            parts.append("N")
        if self.action_first:
            parts.append("AF")
        if self.action_termination:
            parts.append("AT")
        return "-".join(parts)
