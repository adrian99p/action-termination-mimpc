"""Shared Gurobi environment."""

from functools import cache

import gurobipy as gp


@cache
def default_env() -> gp.Env:
    """Return the shared silent Gurobi environment."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env
