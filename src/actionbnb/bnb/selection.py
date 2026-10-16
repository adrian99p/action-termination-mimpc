"""Open nodes: best-bound selection with optional plunging."""

from heapq import heappop, heappush, heappushpop
from itertools import count
from math import inf

from actionbnb.bnb.node import Node


class NodeQueue:
    """Best-bound heap; with rho, the preferred child is processed next while L_p - b <= rho (theta - b)."""

    def __init__(self, rho: float | None = None):
        self.rho = rho
        self.theta = inf
        self._heap: list[tuple[float, int, Node]] = []
        self._count = count()
        self._next: Node | None = None

    def __len__(self) -> int:
        return len(self._heap) + (self._next is not None)

    def push(self, node: Node) -> None:
        heappush(self._heap, (node.bound, next(self._count), node))

    def push_children(self, children: list[Node]) -> None:
        """Children in exploration order; when plunging, the first is held as the preferred child."""
        if self.rho is not None and children:
            self._next = children[0]
            children = children[1:]
        for child in children:
            self.push(child)

    def pop(self) -> Node:
        node, self._next = self._next, None
        if node is None:
            node = heappop(self._heap)[2]
        else:
            b = self._heap[0][0] if self._heap else node.bound
            if self.theta < inf and node.bound - b > self.rho * (self.theta - b):
                node = heappushpop(self._heap, (node.bound, next(self._count), node))[2]
        node.is_open = False
        return node

    def lower_bound(self) -> float:
        """Smallest bound among open nodes; inf if none."""
        bound = self._heap[0][0] if self._heap else inf
        if self._next is not None and self._next.bound < bound:
            return self._next.bound
        return bound
