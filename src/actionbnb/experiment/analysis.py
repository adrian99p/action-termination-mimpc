"""Selecting and aggregating experiment records."""

import numpy as np

RUNTIME_SHIFT = 0.01  # s
NODE_SHIFT = 1.0


def select(records: list[dict], **criteria) -> list[dict]:
    """Records whose fields equal all criteria."""
    return [r for r in records if all(r[key] == value for key, value in criteria.items())]


def sgm(values, shift: float) -> float:
    """Shifted geometric mean exp(mean(log(v + shift))) - shift."""
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float) + shift))) - shift)


def sgm_runtime(records: list[dict]) -> float:
    return sgm([r["runtime"] for r in records], RUNTIME_SHIFT)


def sgm_nodes(records: list[dict]) -> float:
    return sgm([r["nodes"] for r in records], NODE_SHIFT)
