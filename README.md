# Action-Based Termination of Branch-and-Bound for Mixed-Integer MPC

Code for the letter

> A. L. Pavlak, J.-M. Godhavn, and L. S. Imsland, "Action-Based Termination of
> Branch-and-Bound for Mixed-Integer Model Predictive Control," submitted to
> *IEEE Control Systems Letters*, 2026.

It contains the branch-and-bound (B&B) solver, the four benchmark problems, and
the scripts behind the figures, Table I, and the numbers of the results section.

## Example

AF+AT combines action-first branching (AF) with action termination (AT), which
stops once no open node that admits a different action can beat the incumbent.
On the double-integrator benchmark, with most-fractional branching (MF) and deep
plunging (DPLG) that prefers the child nearest the relaxation value:

```python
from dataclasses import replace

from actionbnb.benchmarks import make_benchmark
from actionbnb.bnb import BnBConfig, solve

problem = make_benchmark("integrator", seed=0, N=20).instance()
strategy = BnBConfig(branching="MF", selection="DPLG", child_order="nearest")

full = solve(problem, strategy)  # conventional B&B
afat = solve(problem, replace(strategy, action_first=True, action_termination=True))
for result in (full, afat):
    print(result.status, result.action, result.objective, result.nodes)
```

| Run | Status | Action | Objective | Nodes |
| --- | --- | --- | --- | --- |
| B&B | `optimal` | `(1.0,)` | 51.26 | 4299 |
| AF+AT | `action_satisfied` | `(1.0,)` | 54.59 | 20 |

AF+AT returns the same action after 20 nodes, with an incumbent trajectory that
is still suboptimal.

## Installation

Requires Python 3.12 or newer (the paper used 3.14.8) and a Gurobi license.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # exact versions used for the paper
pip install -e .
```

The size-limited license bundled with `gurobipy` solves the diagnostic, the
integrator, and the buck converter, but not the refrigeration problem for
N ≥ 15, which includes Table I and Fig. 3. Free academic licenses are available
at <https://www.gurobi.com/academia/>.

## Reproducing the results

From the repository root:

```bash
python -m reproduce.reproduce --assets-only  # seconds: outputs from the committed results
python -m reproduce.reproduce --smoke        # fewer seeds, for a quick check
python -m reproduce.reproduce                # about 9 h: the paper's full runs
```

`--assets-only` generates the figures and tables from the committed results,
and the full run recomputes them. `--smoke` runs the whole pipeline with fewer
seeds for a faster test. The figures and Table I are then saved in
[reproduce/results/](reproduce/results/):

| File | Result |
| --- | --- |
| `fig2_bound_traces.png` | Fig. 2 |
| `fig3_horizon_scaling.png` | Fig. 3 |
| `fig4_closed_loop.png` | Fig. 4 |
| `table1_ablation.md` | Table I |

A rerun with the same Gurobi version should give the same node counts and first
inputs. Runtimes depend on the machine (the paper used one thread of an Intel
Core Ultra 7 165U), so compare speedups rather than absolute times.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/actionbnb/bnb/` | B&B solver: competing-action bound `L_{≠â}` in [`actions.py`](src/actionbnb/bnb/actions.py), stopping tests of Theorem 1 (`_termination_status`) and action-first branching (`_branch`) in [`solver.py`](src/actionbnb/bnb/solver.py) |
| `src/actionbnb/benchmarks/` | The four benchmarks, each documented in its module docstring |
| `src/actionbnb/oracle/` | Gurobi reference solves |
| `src/actionbnb/mpc/` | Closed-loop simulation |
| `experiments/` | One script per result |
| `results/` | Raw results and the figures, tables, and numbers computed from them |
| `reproduce/` | `reproduce.py`, which runs every experiment in order, and `results/` with the paper's figures and Table I |

## Contact

Adrian L. Pavlak, Norwegian University of Science and Technology (NTNU),
<adrian.l.pavlak@ntnu.no>
