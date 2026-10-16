"""Compatible actions of nodes and the competing-action bound."""

from heapq import heappop, heappush
from itertools import count
from math import inf

import numpy as np

from actionbnb.action_encoding import ActionEncoding
from actionbnb.bnb.node import Node


class ActionMasks:
    """Encoding patterns and actions compatible with a node, as bit masks."""

    def __init__(self, encoding: ActionEncoding, binary_idx: np.ndarray):
        position = {int(i): j for j, i in enumerate(binary_idx)}
        patterns = list(encoding.patterns)
        self.actions = list(dict.fromkeys(encoding.patterns.values()))
        self.vars = [position[i] for i in encoding.indices]
        self.full = (1 << len(patterns)) - 1
        self._encoding = encoding
        self._index = {a: k for k, a in enumerate(self.actions)}
        self._pattern_action = [1 << self._index[encoding.patterns[p]] for p in patterns]
        self._compat = {
            (j, value): sum(1 << k for k, p in enumerate(patterns) if p[e] == value)
            for e, j in enumerate(self.vars)
            for value in (0, 1)
        }
        self._action_sets: dict[int, int] = {}

    def restrict(self, mask: int, j: int, value: int) -> int:
        compat = self._compat.get((j, value))
        return mask if compat is None else mask & compat

    def action_set(self, mask: int) -> int:
        actions = self._action_sets.get(mask)
        if actions is None:
            actions = 0
            for k, bit in enumerate(self._pattern_action):
                if mask >> k & 1:
                    actions |= bit
            self._action_sets[mask] = actions
        return actions

    def split_var(self, mask: int) -> int | None:
        """First encoding variable whose children both have patterns, while |A_n| > 1."""
        actions = self.action_set(mask)
        if actions & (actions - 1) == 0:
            return None
        compat = self._compat
        for j in self.vars:
            if mask & compat[j, 0] and mask & compat[j, 1]:
                return j
        return None

    def action_index(self, z: np.ndarray) -> int:
        """Index in actions of the first input of an integer-feasible z."""
        return self._index[self._encoding.action(z)]


class ActionTracker:
    """Open nodes in one heap per decided action and one for mixed nodes, giving L_{!=a}."""

    def __init__(self, masks: ActionMasks):
        self.masks = masks
        self._single: list[list] = [[] for _ in masks.actions]
        self._mixed: list = []
        self._count = count()

    def add(self, node: Node) -> None:
        actions = self.masks.action_set(node.mask)
        heap = self._mixed if actions & (actions - 1) else self._single[actions.bit_length() - 1]
        heappush(heap, (node.bound, next(self._count), node))

    def competing_bound(self, action: int) -> float:
        """L_{!=a}: smallest bound among open nodes admitting an action other than a."""
        bound = _top(self._mixed)
        for k, heap in enumerate(self._single):
            if k != action:
                top = _top(heap)
                if top < bound:
                    bound = top
        return bound


def _top(heap: list) -> float:
    while heap and not heap[0][2].is_open:
        heappop(heap)
    return heap[0][0] if heap else inf
