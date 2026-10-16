"""Gurobi reference solver for convex MIQPs."""

from dataclasses import replace
from math import inf
from time import perf_counter

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from actionbnb.gurobi import default_env, status_from_gurobi
from actionbnb.oracle.result import OracleResult
from actionbnb.problem import MIQPProblem
from actionbnb.status import SolveStatus

PARAMS = {"Threads": 1, "Seed": 0, "MIPGap": 0.0, "MIPGapAbs": 0.0, "IntFeasTol": 1e-6, "FeasibilityTol": 1e-9}


def solve_with_oracle(mip: MIQPProblem, time_limit: float = GRB.INFINITY) -> OracleResult:
    """Solve an MIQP with Gurobi, otherwise at default settings; the runtime excludes model construction."""
    with gp.Model(env=default_env()) as model:
        for name, value in PARAMS.items():
            model.setParam(name, value)
        model.setParam("TimeLimit", time_limit)

        vtype = np.full(mip.n, GRB.CONTINUOUS)
        vtype[mip.binary_idx] = GRB.BINARY
        z = model.addMVar(mip.n, lb=mip.lb, ub=mip.ub, vtype=vtype, name="z")
        if mip.A is not None:
            model.addConstr(mip.A @ z == mip.b)
        if mip.G is not None:
            model.addConstr(mip.G @ z <= mip.h)
        model.setObjective(mip.q @ z + mip.constant + 0.5 * z @ mip.P @ z)

        start = perf_counter()
        model.optimize()
        runtime = perf_counter() - start

        solved = model.SolCount > 0 and model.Status != GRB.UNBOUNDED
        return OracleResult(
            status=status_from_gurobi(model.Status),
            objective=model.ObjVal if solved else None,
            z=z.X if solved else None,
            runtime=runtime,
            nodes_explored=int(model.NodeCount) if model.IsMIP else 0,
            n_simplex=int(model.IterCount),
        )


def solve_action_values(mip: MIQPProblem) -> dict[tuple[float, ...], float]:
    """V(a) for every feasible first input a, each from a solve with the encoding fixed to a pattern of a."""
    values: dict[tuple[float, ...], float] = {}
    idx = list(mip.action_encoding.indices)
    for pattern, action in mip.action_encoding.patterns.items():
        lb, ub = mip.lb.copy(), mip.ub.copy()
        lb[idx] = ub[idx] = pattern
        result = solve_with_oracle(replace(mip, lb=lb, ub=ub))
        if result.status == SolveStatus.OPTIMAL:
            values[action] = min(values.get(action, inf), result.objective)
        elif result.status not in (SolveStatus.INFEASIBLE, SolveStatus.INFEASIBLE_OR_UNBOUNDED):
            raise RuntimeError(f"The solve with first input {action} ended with status {result.status}.")
    return values
