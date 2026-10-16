"""Branch and bound with action-based termination and action-first branching."""

from .config import BnBConfig
from .result import BnBResult, BnBTrace
from .solver import solve

__all__ = ["BnBConfig", "BnBResult", "BnBTrace", "solve"]
