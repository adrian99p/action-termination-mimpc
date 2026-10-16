"""Interface and building blocks shared by the benchmarks.

    layout.py    decision-vector layout z = [X, U, Delta, Aux]
    inputs.py    finite input sets and their binary encodings
    builder.py   HorizonBuilder: dynamics, input sets, constraints, and costs
    base.py      the Benchmark interface and exact discretization
"""

from .base import Benchmark, zero_order_hold
from .builder import HorizonBuilder
from .inputs import DiscreteInput, on_off, three_level
from .layout import StageLayout

__all__ = ["Benchmark", "DiscreteInput", "HorizonBuilder", "StageLayout", "on_off", "three_level", "zero_order_hold"]
