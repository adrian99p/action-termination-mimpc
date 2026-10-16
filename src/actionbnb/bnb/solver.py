"""Branch-and-bound with action-based termination and action-first branching."""

from math import inf, nan
from time import perf_counter

import numpy as np

from actionbnb.bnb.actions import ActionMasks, ActionTracker
from actionbnb.bnb.branching import make_brancher
from actionbnb.bnb.config import BnBConfig
from actionbnb.bnb.node import Node, QPSolution
from actionbnb.bnb.relaxation import NodeQP
from actionbnb.bnb.result import BnBResult, BnBTrace
from actionbnb.bnb.selection import NodeQueue
from actionbnb.problem import MIQPProblem
from actionbnb.status import SolveStatus


class BnBSolver:
    """Branch-and-bound solver."""

    def __init__(self, problem: MIQPProblem, config: BnBConfig | None = None):
        self.problem = problem
        self.config = BnBConfig() if config is None else config

    def _init_search(self) -> None:
        """Initialize the search."""
        problem, cfg = self.problem, self.config
        encoding = problem.action_encoding
        track = encoding is not None and (cfg.action_termination or cfg.trace)

        self.qp = NodeQP(problem)
        self.masks = ActionMasks(encoding, problem.binary_idx) if cfg.action_first or track else None
        self.tracker = ActionTracker(self.masks) if track else None
        self.brancher = make_brancher(cfg.branching, problem.n_binary)
        self.queue = NodeQueue(cfg.rho if cfg.selection == "DPLG" else None)
        self.trace = BnBTrace() if cfg.trace else None

        self.incumbent_z: np.ndarray | None = None
        self.incumbent_obj = inf
        self.incumbent_action: int | None = None
        self.theta = inf
        self.nodes = 0
        self.first_incumbent: tuple[int, float] | None = None
        self.certified: tuple[int, float] | None = None

        self.start = perf_counter()
        self.deadline = inf if cfg.time_limit is None else self.start + cfg.time_limit
        self._push([Node.root(problem.n_binary, None if self.masks is None else self.masks.full)])

    def _search(self) -> BnBResult:
        """Run the branch-and-bound search."""
        qp, queue, brancher = self.qp, self.queue, self.brancher
        binary_idx, int_tol = self.problem.binary_idx, self.config.int_tol
        while True:

            # 0. Termination checks
            status = self._termination_status()
            if status is not None:
                break

            # 1. Select a node; skip it if a later incumbent beats its bound
            node = queue.pop()
            if node.bound >= self.theta:
                continue
            self.nodes += 1

            # 2. Solve the node QP
            solution = node.solution
            if solution is None:
                qp_status, solution = qp.solve(node)
                if qp_status is SolveStatus.INFEASIBLE or qp_status is SolveStatus.CUTOFF:
                    continue
                if qp_status is not SolveStatus.OPTIMAL:
                    node.is_open = True  # Keep its region in the returned bounds.
                    self._push([node])
                    status = qp_status
                    break
                if node.branch_var >= 0:
                    brancher.update(node, solution.objective)

            # 3. Prune by bound
            if solution.objective >= self.theta:
                continue

            # 4. Update the incumbent if the solution is integral
            x = solution.z[binary_idx]
            candidates = np.flatnonzero(np.abs(x - np.round(x)) > int_tol)
            if candidates.size == 0:
                self._update_incumbent(solution.objective, solution.z)
                continue

            # 5. Branch and queue the children
            self._push(self._branch(node, solution, x, candidates))

        return self._result(status)

    def _termination_status(self) -> SolveStatus | None:
        """Stopping tests."""
        lower = min(self.queue.lower_bound(), self.incumbent_obj)
        competing = self._competing_bound()
        if self.trace is not None:
            self._record(lower, competing)

        if self.incumbent_z is None:
            if not self.queue:
                return SolveStatus.INFEASIBLE
        else:
            certified = competing >= self.theta
            if certified and self.certified is None:
                self.certified = (self.nodes, perf_counter() - self.start)
            if lower >= self.theta:
                return SolveStatus.OPTIMAL
            if certified and self.config.action_termination:
                return SolveStatus.ACTION_SATISFIED

        if perf_counter() >= self.deadline:
            return SolveStatus.TIME_LIMIT
        return None

    def _branch(self, node: Node, solution: QPSolution, x: np.ndarray, candidates: np.ndarray) -> list[Node]:
        """Children in exploration order: action-first while the node admits several actions, else the base rule."""
        j = self.masks.split_var(node.mask) if self.config.action_first else None
        action_first = j is not None
        if not action_first:
            j = self.brancher.select(x, candidates)
        if solution.basis is None:
            solution.basis = self.qp.basis()

        xj = x[j]
        up_first = self.config.child_order == "up" or xj >= 0.5
        children = []
        for value in (1, 0) if up_first else (0, 1):
            mask = None if self.masks is None else self.masks.restrict(node.mask, j, value)
            if mask == 0:
                continue
            child = node.child(j, value, solution.objective, solution.basis, abs(xj - value), mask)
            if action_first and child.branch_dist <= self.config.int_tol:
                child.solution = solution
            children.append(child)
        return children

    def _push(self, children: list[Node]) -> None:
        self.queue.push_children(children)
        if self.tracker is not None:
            for child in children:
                self.tracker.add(child)

    def _update_incumbent(self, objective: float, z: np.ndarray) -> None:
        self.incumbent_z, self.incumbent_obj = z, objective
        self.incumbent_action = None if self.masks is None else self.masks.action_index(z)
        self.theta = objective - self.config.eps - self.config.bound_tol
        self.queue.theta = self.theta
        self.qp.set_cutoff(self.theta)
        if self.first_incumbent is None:
            self.first_incumbent = (self.nodes, perf_counter() - self.start)

    def _competing_bound(self) -> float:
        """L_{!=a} for the incumbent action; nan without tracking or incumbent."""
        if self.tracker is None or self.incumbent_action is None:
            return nan
        return self.tracker.competing_bound(self.incumbent_action)

    def _record(self, lower: float, competing: float) -> None:
        self.trace.nodes.append(self.nodes)
        self.trace.upper.append(self.incumbent_obj)
        self.trace.lower.append(lower)
        self.trace.competing.append(competing)

    def _result(self, status: SolveStatus) -> BnBResult:
        encoding = self.problem.action_encoding
        z = self.incumbent_z
        first, certified = self.first_incumbent or (None, None), self.certified or (None, None)
        return BnBResult(
            status=status,
            z=z,
            objective=self.incumbent_obj,
            action=None if z is None or encoding is None else encoding.action(z),
            lower_bound=min(self.queue.lower_bound(), self.incumbent_obj),
            competing_bound=None if self.tracker is None else self._competing_bound(),
            runtime=perf_counter() - self.start,
            nodes=self.nodes,
            qp_solves=self.qp.n_solves,
            qp_iterations=self.qp.n_iterations,
            qp_time=self.qp.time,
            first_incumbent_node=first[0],
            first_incumbent_time=first[1],
            certified_node=certified[0],
            certified_time=certified[1],
            trace=self.trace,
        )

    def solve(self) -> BnBResult:
        """Run branch and bound."""
        try:
            self._init_search()
            return self._search()
        finally:
            self.qp.close()


def solve(problem: MIQPProblem, config: BnBConfig | None = None) -> BnBResult:
    """Solve problem by branch and bound."""
    return BnBSolver(problem, config).solve()
