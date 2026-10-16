"""Variable selection among fractional binaries: FF, MF, and PC."""

import numpy as np

from actionbnb.bnb.node import Node

EPS = 1e-6


class Brancher:
    """Selects a branching binary from the fractional candidates of a node."""

    def __init__(self, n_binary: int):
        self.n_binary = n_binary

    def select(self, x: np.ndarray, candidates: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, node: Node, objective: float) -> None:
        """Observe the solved QP objective of a branched child."""


class FirstFractional(Brancher):
    """Lowest-index fractional binary; binaries are stored stage by stage, so the earliest stage first."""

    def select(self, x: np.ndarray, candidates: np.ndarray) -> int:
        return int(candidates[0])


class MostFractional(Brancher):
    """Fractional binary closest to 0.5."""

    def select(self, x: np.ndarray, candidates: np.ndarray) -> int:
        return int(candidates[np.argmin(np.abs(x[candidates] - 0.5))])


class Pseudocost(Brancher):
    """Product of predicted down and up gains; equals MF before any observation."""

    def __init__(self, n_binary: int):
        super().__init__(n_binary)
        self._psi = np.ones((2, n_binary))
        self._count = np.zeros((2, n_binary), dtype=np.int64)
        self._unseen = np.ones((2, n_binary), dtype=bool)
        self._total = [0.0, 0.0]
        self._n = [0, 0]

    def update(self, node: Node, objective: float) -> None:
        j, d = node.branch_var, int(node.fixed[node.branch_var])
        gain = max(objective - node.bound, 0.0) / node.branch_dist
        self._total[d] += gain
        self._n[d] += 1
        count = self._count[d, j] + 1
        self._count[d, j] = count
        if count == 1:
            self._unseen[d, j] = False
            self._psi[d, j] = gain
        else:
            self._psi[d, j] += (gain - self._psi[d, j]) / count
        self._psi[d, self._unseen[d]] = self._total[d] / self._n[d]

    def select(self, x: np.ndarray, candidates: np.ndarray) -> int:
        f = x[candidates]
        psi = self._psi[:, candidates]
        score = np.maximum(f * psi[0], EPS) * np.maximum((1.0 - f) * psi[1], EPS)
        return int(candidates[score.argmax()])


BRANCHERS = {"FF": FirstFractional, "MF": MostFractional, "PC": Pseudocost}


def make_brancher(rule: str, n_binary: int) -> Brancher:
    return BRANCHERS[rule](n_binary)
