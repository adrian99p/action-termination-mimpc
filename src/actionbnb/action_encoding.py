"""Binary encodings of complete first inputs."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ActionEncoding:
    """Maps admissible binary patterns to physical inputs."""

    indices: tuple[int, ...]
    patterns: dict[tuple[int, ...], tuple[float, ...]]

    def action(self, z: np.ndarray) -> tuple[float, ...]:
        """Decode the complete first input of an integer-feasible solution."""
        pattern = tuple(int(round(z[j])) for j in self.indices)
        return self.patterns[pattern]
