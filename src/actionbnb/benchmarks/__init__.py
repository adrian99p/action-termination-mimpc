"""MI-MPC benchmark problems.

Benchmarks:
    buck.py              step-down DC-DC converter adapted from Geyer et al. (2008)
    diagnostic.py        constructed action-certification diagnostic (open loop)
    integrator.py        double integrator with three-level acceleration
    refrigeration.py     supermarket display cases adapted from Yang (2017)

Each benchmark is a frozen dataclass implementing Benchmark: a nominal linear
plant, a seeded initial condition, and the horizon MIQP for a measured state.
The same object serves open-loop experiments (instance()) and closed-loop
simulation (problem() and step()). The interface and building blocks live in
core/.
"""

from .buck import Buck
from .core import Benchmark
from .diagnostic import Diagnostic
from .integrator import Integrator
from .refrigeration import Refrigeration

BENCHMARKS: dict[str, type[Benchmark]] = {
    cls.name: cls for cls in (Buck, Diagnostic, Integrator, Refrigeration)
}


def make_benchmark(name: str, **params) -> Benchmark:
    """Instantiate a registered benchmark; params (e.g. seed, N) override its defaults."""
    try:
        cls = BENCHMARKS[name]
    except KeyError:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}.") from None
    return cls(**params)


__all__ = [
    "BENCHMARKS", "Benchmark", "Buck", "Diagnostic", "Integrator", "Refrigeration", "make_benchmark",
]
