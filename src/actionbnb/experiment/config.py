"""Definition of a computational experiment."""

from dataclasses import asdict, dataclass

from actionbnb.bnb import BnBConfig


@dataclass(frozen=True)
class Experiment:
    """Solve every benchmark instance with Gurobi and each B&B configuration.

    benchmarks maps each benchmark name to its parameters other than the seed,
    e.g. {"integrator": {"N": 20}}.

    Each solve is repeated, after an optional discarded warm-up, and the run with
    the median runtime is kept.
    """

    name: str
    benchmarks: dict[str, dict]
    configs: tuple[BnBConfig, ...]
    seeds: tuple[int, ...] = tuple(range(20))
    repeats: int = 3
    warmup: bool = True

    def to_dict(self) -> dict:
        spec = asdict(self)
        spec["configs"] = [{"name": c.name, **asdict(c)} for c in self.configs]
        return spec
