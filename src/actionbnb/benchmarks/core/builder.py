"""Assemble the horizon MIQP of a linear MI-MPC benchmark from standard terms.

Example, the integrator benchmark:

    builder = HorizonBuilder(StageLayout(N, n_x=2, n_u=1, n_delta=2))
    builder.dynamics(A, B, x0)
    builder.discrete_input(three_level())
    builder.tracking_cost(Q)
    builder.input_cost(R)
    problem = builder.build("integrator")

The builder starts with free states, inputs, and auxiliaries, binaries in
[0, 1], no constraints, and zero cost. Each call appends constraint rows or
cost terms in call order, so a benchmark reads like its formulation. Costs
follow the MIQPProblem convention 0.5 z'Pz + q'z + constant, and weight
matrices must be symmetric positive semidefinite.

References and weights broadcast over stages: a reference is either one
vector, used at every stage, or one row per stage, and a weight or input
matrix B is either one matrix or one per stage.
"""

import numpy as np
from scipy import sparse as sp

from actionbnb.benchmarks.core.inputs import DiscreteInput
from actionbnb.benchmarks.core.layout import StageLayout, shifted, stages
from actionbnb.problem import MIQPProblem


class HorizonBuilder:
    """Accumulates the constraints and cost of one horizon MIQP."""

    def __init__(self, layout: StageLayout):
        self.layout = layout
        n = layout.n_z
        self.lb = layout.vector(x=-np.inf, u=-np.inf, delta=0.0, aux=-np.inf)
        self.ub = layout.vector(x=np.inf, u=np.inf, delta=1.0, aux=np.inf)
        self._P = sp.csr_array((n, n))
        self._q = np.zeros(n)
        self._constant = 0.0
        self._equalities: list[tuple[sp.csr_array, np.ndarray]] = []
        self._inequalities: list[tuple[sp.csr_array, np.ndarray]] = []
        self._inputs: DiscreteInput | None = None

    # Constraints

    def dynamics(self, A, B, x0) -> None:
        """x_{i+1} = A x_i + B_i u_i for i = 0, ..., N-1, from the measured x_0."""
        layout, N = self.layout, self.layout.N
        rows = layout.columns(x=sp.eye_array(N * layout.n_x) - shifted(A, N), u=-stages(B, N))
        rhs = np.zeros(N * layout.n_x)
        rhs[:layout.n_x] = np.asarray(A) @ np.asarray(x0, dtype=float)
        self._equalities.append((rows, rhs))

    def discrete_input(self, inputs: DiscreteInput) -> None:
        """u_i = S delta_i with G delta_i <= h at every stage; also sets the first-input encoding."""
        layout, N = self.layout, self.layout.N
        if (inputs.n_u, inputs.n_delta) != (layout.n_u, layout.n_delta):
            raise ValueError(
                f"Input set has n_u={inputs.n_u}, n_delta={inputs.n_delta}; "
                f"layout has n_u={layout.n_u}, n_delta={layout.n_delta}."
            )
        linkage = layout.columns(u=stages(np.eye(layout.n_u), N), delta=-stages(inputs.S, N))
        self._equalities.append((linkage, np.zeros(N * layout.n_u)))
        if inputs.G.shape[0] > 0:
            self.stage_inequalities(inputs.h, delta=inputs.G)
        self.bounds("u", *inputs.bounds())
        self._inputs = inputs

    def stage_inequalities(self, h, x=None, u=None, delta=None, aux=None) -> None:
        """G_x x_{i+1} + G_u u_i + G_delta delta_i + G_aux aux_i <= h at every stage."""
        N = self.layout.N
        rows = self.layout.columns(*(None if block is None else stages(block, N) for block in (x, u, delta, aux)))
        self._inequalities.append((rows, np.tile(np.asarray(h, dtype=float), N)))

    def bounds(self, group: str, lb, ub) -> None:
        """Set the bounds of a group, per stage or for all stages."""
        shape = (self.layout.N, self.layout.size(group))
        idx = self.layout.index(group)
        self.lb[idx] = np.broadcast_to(np.asarray(lb, dtype=float), shape).ravel()
        self.ub[idx] = np.broadcast_to(np.asarray(ub, dtype=float), shape).ravel()

    # Costs

    def tracking_cost(self, Q, ref=None, C=None) -> None:
        """sum_{i=1}^{N} (C x_i - r_i)' Q (C x_i - r_i); C defaults to the identity."""
        self._quadratic(self.layout.index("x"), Q, ref, C)

    def terminal_cost(self, Q, ref=None, C=None) -> None:
        """(C x_N - r)' Q (C x_N - r); C defaults to the identity."""
        self._quadratic(self.layout.index("x", self.layout.N - 1), Q, ref, C)

    def input_cost(self, R, ref=None) -> None:
        """sum_{i=0}^{N-1} (u_i - r_i)' R (u_i - r_i)."""
        self._quadratic(self.layout.index("u"), R, ref, None)

    def linear_input_cost(self, c) -> None:
        """sum_{i=0}^{N-1} c' u_i, e.g. the energy used by on/off inputs."""
        shape = (self.layout.N, self.layout.n_u)
        self._q[self.layout.index("u")] += np.broadcast_to(np.asarray(c, dtype=float), shape).ravel()

    def zone_cost(self, Q, width, ref=None, C=None) -> None:
        """sum_{i=1}^{N} e_i' Q e_i, with e_i the excess of C x_i - r outside the box [-width, width].

        Zone tracking: no cost inside the box. Uses the auxiliaries as e_i, with
        |C x_i - r - e_i| <= width componentwise, so the layout needs n_aux = rows of C.
        """
        layout = self.layout
        C = np.eye(layout.n_x) if C is None else np.atleast_2d(np.asarray(C, dtype=float))
        if layout.n_aux != C.shape[0]:
            raise ValueError(f"zone_cost needs n_aux = {C.shape[0]}, got n_aux = {layout.n_aux}.")
        ref = np.broadcast_to(np.asarray(0.0 if ref is None else ref, dtype=float), C.shape[0])
        eye = np.eye(C.shape[0])
        self.stage_inequalities(ref + width, x=C, aux=-eye)
        self.stage_inequalities(width - ref, x=-C, aux=eye)
        self._quadratic(layout.index("aux"), Q, None, None)

    def switching_cost(self, weight, u_prev) -> None:
        """sum_{i=0}^{N-1} weight' |u_i - u_{i-1}|, with u_{-1} = u_prev the previously applied input.

        Uses the auxiliaries as epigraph variables, aux_i >= |u_i - u_{i-1}|, so the
        layout needs n_aux = n_u.
        """
        layout, N = self.layout, self.layout.N
        if layout.n_aux != layout.n_u:
            raise ValueError(f"switching_cost needs n_aux = n_u = {layout.n_u}, got n_aux = {layout.n_aux}.")
        eye = np.eye(layout.n_u)
        diff = stages(eye, N) - shifted(eye, N)  # u_i - u_{i-1}, without u_{-1}
        rhs = np.zeros(N * layout.n_u)
        rhs[:layout.n_u] = u_prev  # u_{-1} moves to the right-hand side
        aux = -stages(eye, N)
        self._inequalities.append((layout.columns(u=diff, aux=aux), rhs))
        self._inequalities.append((layout.columns(u=-diff, aux=aux), -rhs))
        self.bounds("aux", 0.0, np.inf)
        self._q[layout.index("aux")] += np.broadcast_to(np.asarray(weight, dtype=float), (N, layout.n_u)).ravel()

    def _quadratic(self, idx: np.ndarray, Q, ref, C) -> None:
        """Add sum_j (C v_j - r_j)' Q_j (C v_j - r_j) over consecutive blocks v_j of z[idx]."""
        Q = np.asarray(Q, dtype=float)
        n_y = Q.shape[-1] if Q.ndim else 1
        C = np.eye(n_y) if C is None else np.atleast_2d(np.asarray(C, dtype=float))
        n_blocks = idx.size // C.shape[1]
        Q = np.broadcast_to(Q if Q.ndim == 3 else Q.reshape(n_y, n_y), (n_blocks, n_y, n_y))
        ref = np.broadcast_to(np.asarray(0.0 if ref is None else ref, dtype=float), (n_blocks, n_y))

        # (Cv - r)'Q(Cv - r) = 0.5 v'(2 C'QC)v - 2 r'QC v + r'Q r
        hessian = sp.coo_array(stages(2.0 * C.T @ Q @ C, n_blocks))
        n = self.layout.n_z
        self._P = self._P + sp.csr_array((hessian.data, (idx[hessian.row], idx[hessian.col])), shape=(n, n))
        self._q[idx] += (-2.0 * np.einsum("bi,bij->bj", ref, Q) @ C).ravel()
        self._constant += float(np.einsum("bi,bij,bj->", ref, Q, ref))

    def build(self, name: str) -> MIQPProblem:
        """Return the assembled MIQP; the first-input encoding comes from discrete_input()."""

        def stack(blocks):
            if not blocks:
                return None, None
            return sp.vstack([rows for rows, _ in blocks], format="csr"), np.concatenate([rhs for _, rhs in blocks])

        A, b = stack(self._equalities)
        G, h = stack(self._inequalities)
        return MIQPProblem(
            P=self._P.tocsr(),
            q=self._q.copy(),
            lb=self.lb.copy(),
            ub=self.ub.copy(),
            binary_idx=self.layout.binary_idx,
            A=A,
            b=b,
            G=G,
            h=h,
            constant=self._constant,
            name=name,
            action_encoding=None if self._inputs is None else self._inputs.encoding(self.layout),
        )
