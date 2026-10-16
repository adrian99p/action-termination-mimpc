"""Supermarket display cases with on/off inlet valves, adapted from Yang.

CASES display cases stand in a row. In case j, the goods temperature Tg_j and the
air temperature Ta_j follow

    m_g,j c_g dTg_j/dt = K_GA (Ta_j - Tg_j),
    m_a c_a dTa_j/dt   = K_GA (Tg_j - Ta_j) + K_AMB (T_AMB - Ta_j) - K_EVAP (Ta_j - T_EVAP u_j)
                         + K_NB sum_{l neighbour of j} (Ta_l - Ta_j),

where u_j in {0, 1} closes or opens the inlet (expansion) valve. The state x stacks
[Tg; Ta] in deviations from the all-valves-closed equilibrium T_CLOSED, which makes
the dynamics linear. The MPC problem at sampling instant k is

    min  sum_{i=0}^{N-1} e_{i+1}' Q e_{i+1} + C_ENERGY 1'u_i
    s.t. x_{i+1} = A x_i + B u_i   (exact zero-order hold, DT = 60 s),
         |x_{i+1} - x_mid - e_{i+1}| <= (T_MAX - T_MIN) / 2   componentwise,
         u_i in {0, 1}^CASES,

where e is the excess of every temperature outside the band [T_MIN, T_MAX] (x_mid is
its midpoint), Q = blkdiag(Q_GOODS I, Q_AIR I), and C_ENERGY prices each open valve
per sample. The seed samples the food masses and the state at the end of a defrost:
from temperatures uniform in the band, all valves stay closed for 10 to 20 min, and
the controller pulls the cases back into the band.

Source:
    I. Yang, "Online combinatorial optimization for interconnected refrigeration
    systems: Linear approximation and submodularity," IFAC-PapersOnLine,
    50(1):7389-7396, 2017. The model and its parameters, the fit of T_EVAP at the
    suction pressure 1.4 bar, the band [T_MIN, T_MAX] = [0, 5] C, and the +-20 %
    food-mass perturbation are taken from there.

Adaptations:
    Three of Yang's ten cases are used, and the one-step objective is replaced by
    an N-step MPC with a zone cost (Ferramosca et al., 2010) on the goods and air
    temperatures and a price C_ENERGY per open valve. The suction pressure is held
    at its reference. The weights and the defrost start are chosen here.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from actionbnb.benchmarks.core import Benchmark, HorizonBuilder, StageLayout, on_off, zero_order_hold
from actionbnb.problem import MIQPProblem

# Display-case parameters in SI units, from Yang.
M_GOODS = 200.0  # kg, nominal
C_GOODS = 1000.0  # J/(kg K)
M_AIR = 50.0  # kg
C_AIR = 1000.0  # J/(kg K)
K_GA = 300.0  # W/K, goods to air
K_EVAP = 225.0  # W/K, air to evaporator
K_NB = 500.0  # W/K, air to neighbouring air
K_AMB = 275.0  # W/K, ambient to air
T_AMB = 20.0  # C
P_SUC = 1.4  # bar, suction pressure reference
T_EVAP = -4.3544 * P_SUC**2 + 29.2240 * P_SUC - 51.2005  # C, R134a
T_MIN, T_MAX = 0.0, 5.0  # C
MASS_SPREAD = 0.2
DT = 60.0  # s

CASES = 3
Q_GOODS = 1.0
Q_AIR = 0.3
C_ENERGY = 0.1
DEFROST = (10, 20)  # samples with all valves closed before k = 0
T_CLOSED = K_AMB * T_AMB / (K_AMB + K_EVAP)  # equilibrium with all valves closed
INPUTS = on_off(CASES)


@dataclass(frozen=True)
class Refrigeration(Benchmark):
    name: ClassVar[str] = "refrigeration"
    seed: int = 0
    N: int = 20

    @property
    def layout(self) -> StageLayout:
        # One auxiliary per temperature: its excess outside the band.
        return StageLayout(self.N, n_x=2 * CASES, n_u=CASES, n_delta=CASES, n_aux=2 * CASES)

    def _goods_mass(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return M_GOODS * rng.uniform(1 - MASS_SPREAD, 1 + MASS_SPREAD, CASES)

    def dynamics(self, k: int = 0) -> tuple[np.ndarray, np.ndarray]:
        neighbours = K_NB * (np.eye(CASES, k=1) + np.eye(CASES, k=-1))
        air = neighbours - np.diag(neighbours.sum(axis=1) + K_GA + K_AMB + K_EVAP)
        goods = K_GA / (self._goods_mass() * C_GOODS)
        cap_air = M_AIR * C_AIR
        A_c = np.block([
            [np.diag(-goods), np.diag(goods)],
            [K_GA / cap_air * np.eye(CASES), air / cap_air],
        ])
        B_c = np.vstack([np.zeros((CASES, CASES)), K_EVAP * T_EVAP / cap_air * np.eye(CASES)])
        return zero_order_hold(A_c, B_c, DT)

    def initial_state(self) -> np.ndarray:
        rng = np.random.default_rng([self.seed, 1])
        x = rng.uniform(T_MIN, T_MAX, 2 * CASES) - T_CLOSED
        A, _ = self.dynamics()
        return np.linalg.matrix_power(A, rng.integers(DEFROST[0], DEFROST[1] + 1)) @ x

    def problem(self, x: np.ndarray, k: int = 0, u_prev: np.ndarray | None = None) -> MIQPProblem:
        """Time-invariant and memoryless: k only labels the problem, and u_prev is unused."""
        A, B = self.dynamics()
        builder = HorizonBuilder(self.layout)
        builder.dynamics(A, B, x)
        builder.discrete_input(INPUTS)
        Q = np.diag(np.r_[np.full(CASES, Q_GOODS), np.full(CASES, Q_AIR)])
        builder.zone_cost(Q, (T_MAX - T_MIN) / 2, ref=(T_MAX + T_MIN) / 2 - T_CLOSED)
        builder.linear_input_cost(C_ENERGY)
        return builder.build(f"{self.name}_seed{self.seed}_N{self.N}_k{k}")
