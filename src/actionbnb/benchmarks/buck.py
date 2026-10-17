"""Synchronous step-down (buck) DC-DC converter with direct switching, adapted from Geyer et al.

The primary switch S1 is closed (u = 1) or open (u = 0), and the synchronous
switch S2 is operated dually, so the converter stays in continuous conduction
and is linear. With inductor current and output voltage x = [i_l, v_o] in per
unit and the input voltage v_s, the MPC problem at sampling instant k is

    min  sum_{i=0}^{N-1} Q_V (v_o,i+1 - V_REF)^2 + LAMBDA |u_i - u_{i-1}|
    s.t. x_{i+1} = A x_i + B v_s u_i   (exact zero-order hold, DT = T_s / 3),
         |i_l,i+1| <= I_MAX,  u_i in {0, 1},

where u_{-1} is the previously applied input. The converter operates at its
nominal output voltage V_0 when the output voltage reference steps down to
V_REF: x_0 = [V_0 / R_O, V_0] and u_{-1} = 0. The seed samples v_s.

Source:
    T. Geyer, G. Papafotiou, and M. Morari, "Hybrid model predictive control
    of the step-down DC-DC converter," IEEE Trans. Control Syst. Technol.,
    16(6):1112-1124, 2008. The circuit model and parameters, the current limit
    I_MAX, the voltage weight Q_V, the nominal output voltage V_0, the subperiod
    T_s / 3, and the input-voltage range are taken from there. One time unit is
    one switching period T_s.

Adaptations:
    The source optimizes a duty cycle per switching period. Here the switch
    position is chosen for every subperiod (direct switching), which makes the
    model exact and linear, and the voltage error is squared. The switching
    weight LAMBDA and the reference step from V_0 to V_REF are chosen here, and
    the load is nominal and known.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from actionbnb.benchmarks.core import Benchmark, HorizonBuilder, StageLayout, on_off, zero_order_hold
from actionbnb.problem import MIQPProblem

# Circuit parameters in per unit, from the source.
X_C = 10.294
X_L = 0.477
R_C = 0.001
R_L = 0.05
R_O = 1.0
I_MAX = 3.0
DT = 1.0 / 3.0  # subperiod, in switching periods
V_S_RANGE = (1.2, 3.0)

# Scenario and objective.
V_0 = 1.0
V_REF = 0.5
Q_V = 4.0
LAMBDA = 0.01
OUTPUT = np.array([[0.0, 1.0]])  # v_o = OUTPUT x
INPUTS = on_off()

# Continuous-time dynamics dx/dt = A_C x + B_C v_s u.
A_C = np.array([
    [-R_L / X_L, -1.0 / X_L],
    [R_O * (X_L - R_C * R_L * X_C) / ((R_O + R_C) * X_C * X_L), -(X_L + R_C * R_O * X_C) / ((R_O + R_C) * X_C * X_L)],
])
B_C = np.array([[1.0 / X_L], [R_O * R_C / ((R_O + R_C) * X_L)]])


A, B = zero_order_hold(A_C, B_C, DT)


@dataclass(frozen=True)
class Buck(Benchmark):
    name: ClassVar[str] = "buck"
    seed: int = 0
    N: int = 30

    @property
    def layout(self) -> StageLayout:
        # One auxiliary per stage: the epigraph variable of |u_i - u_{i-1}|.
        return StageLayout(self.N, n_x=2, n_u=1, n_delta=INPUTS.n_delta, n_aux=1)

    @property
    def input_voltage(self) -> float:
        """Input voltage v_s, sampled from the seed."""
        return float(np.random.default_rng(self.seed).uniform(*V_S_RANGE))

    def dynamics(self, k: int = 0) -> tuple[np.ndarray, np.ndarray]:
        return A, self.input_voltage * B

    def initial_state(self) -> np.ndarray:
        return np.array([V_0 / R_O, V_0])

    def problem(self, x: np.ndarray, k: int = 0, u_prev: np.ndarray | None = None) -> MIQPProblem:
        u_prev = self.initial_input() if u_prev is None else u_prev
        builder = HorizonBuilder(self.layout)
        builder.dynamics(*self.dynamics(k), x)
        builder.discrete_input(INPUTS)
        builder.stage_inequalities([I_MAX, I_MAX], x=[[1.0, 0.0], [-1.0, 0.0]])
        builder.tracking_cost([[Q_V]], ref=[V_REF], C=OUTPUT)
        builder.switching_cost(LAMBDA, u_prev)
        return builder.build(f"{self.name}_seed{self.seed}_N{self.N}_k{k}")
