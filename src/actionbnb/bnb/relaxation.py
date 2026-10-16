"""Node QP relaxations, solved with Gurobi's dual simplex."""

from time import perf_counter

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from actionbnb.bnb.node import FREE, Basis, Node, QPSolution
from actionbnb.gurobi import default_env, status_from_gurobi
from actionbnb.gurobi.capi import CModel
from actionbnb.problem import MIQPProblem
from actionbnb.status import SolveStatus

PARAMS = {"Method": 1, "Presolve": 0, "Threads": 1, "FeasibilityTol": 1e-9, "OptimalityTol": 1e-9}


class NodeQP:
    """Continuous relaxation of an MIQP, re-solved for each node with its binary bounds."""

    def __init__(self, problem: MIQPProblem):
        binary = problem.binary_idx
        lb, ub = problem.lb.copy(), problem.ub.copy()
        lb[binary] = np.maximum(lb[binary], 0.0)
        ub[binary] = np.minimum(ub[binary], 1.0)

        self.model = self._build_model(problem, lb, ub)
        self._c = CModel(self.model)

        self._binary = binary.astype(np.int32)
        self._free_lb = lb[binary]
        self._free_ub = ub[binary]
        self._fixed = np.full(binary.size, FREE, dtype=np.int8)
        self._basis: Basis | None = None

        self.n_solves = 0
        self.n_iterations = 0
        self.time = 0.0

    @staticmethod
    def _build_model(problem: MIQPProblem, lb: np.ndarray, ub: np.ndarray) -> gp.Model:
        """Continuous QP with the given variable bounds."""
        model = gp.Model(env=default_env())
        for name, value in PARAMS.items():
            model.setParam(name, value)
        z = model.addMVar(problem.n, lb=lb, ub=ub)
        if problem.A is not None:
            model.addConstr(problem.A @ z == problem.b)
        if problem.G is not None:
            model.addConstr(problem.G @ z <= problem.h)
        model.setObjective(0.5 * (z @ problem.P @ z) + problem.q @ z + problem.constant, GRB.MINIMIZE)
        model.update()
        return model

    def solve(self, node: Node) -> tuple[SolveStatus, QPSolution | None]:
        start = perf_counter()
        c = self._c
        changed = np.flatnonzero(node.fixed != self._fixed)
        if changed.size:
            k = changed.size
            values = node.fixed[changed]
            free = values == FREE
            c.bound_idx[:k] = self._binary[changed]
            c.bound_lb[:k] = np.where(free, self._free_lb[changed], values)
            c.bound_ub[:k] = np.where(free, self._free_ub[changed], values)
            c.set_bounds(k)
            self._fixed = node.fixed
        # A child of the last solved node starts from the basis Gurobi still holds.
        if node.basis is not None and node.basis is not self._basis:
            c.set_basis(node.basis)
        c.optimize()
        self._basis = None

        status = status_from_gurobi(c.status())
        solution = QPSolution(c.obj_val(), c.get_x()) if status == SolveStatus.OPTIMAL else None
        self.n_solves += 1
        self.n_iterations += c.iter_count()
        self.time += perf_counter() - start
        return status, solution

    def basis(self) -> Basis:
        """Optimal basis of the last solve."""
        self._basis = self._c.get_basis()
        return self._basis

    def set_cutoff(self, value: float) -> None:
        self._c.set_cutoff(value)

    def close(self) -> None:
        self.model.dispose()
