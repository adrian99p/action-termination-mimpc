"""Closed loops of the buck converter: the numbers quoted for them in the text.

The paper's closed loop uses the integrator, where action termination helps during braking
but not near the origin. This script runs the same three controllers as closed_loop.py
(Gurobi, B&B, and AF+AT, both MF-DPLG-N, independent loops with fresh searches and a 60 s
limit) on the buck converter and refrigeration for several seeds, and along the Gurobi loop
records the action values V(a), the relative margin between the best and second-best first
input (ties), and the competing relaxation gap eta of relaxation_gap.py.

    python -m experiments.closed_loop_benchmarks run [--benchmark NAME ...] [--steps 30] [--seeds 0 1 2 3 4]   # about 20 min
    python -m experiments.closed_loop_benchmarks assets                                                        # numbers

Timing: one solve per instant, without repeats, as in closed_loop.py. The paper's results contain the
buck converter only: the refrigeration loops were stopped (see meta.json), because near steady state
their solves approach the 60 s limits. run therefore defaults to the buck converter.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from actionbnb.benchmarks import make_benchmark
from actionbnb.bnb import BnBConfig
from actionbnb.experiment import environment, load, save
from actionbnb.experiment.latex import fixed, write_macros
from actionbnb.mpc import bnb, gurobi, simulate
from actionbnb.oracle import solve_with_oracle
from actionbnb.status import SolveStatus
from experiments.relaxation_gap import competing_relaxation_gap

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "results" / "closed_loop_benchmarks"
SOURCE = "experiments/closed_loop_benchmarks.py"
TIME_LIMIT = 60.0
STRATEGY = BnBConfig(branching="MF", selection="DPLG", child_order="nearest", time_limit=TIME_LIMIT)
AFAT = replace(STRATEGY, action_first=True, action_termination=True)
BENCHMARKS = {"buck": {"N": 30}, "refrigeration": {"N": 20}}
DEFAULT = ["buck"]  # The paper's loops; refrigeration was stopped for time.
EARLY = 4  # First instants after the reference step, reported separately in the text.


def action_values(problem) -> dict | None:
    """V(a) per first input from action-fixed Gurobi solves with the time limit; None if one is unresolved."""
    values = {}
    idx = list(problem.action_encoding.indices)
    for pattern, action in problem.action_encoding.patterns.items():
        lb, ub = problem.lb.copy(), problem.ub.copy()
        lb[idx] = ub[idx] = pattern
        r = solve_with_oracle(replace(problem, lb=lb, ub=ub), time_limit=TIME_LIMIT)
        if r.status == SolveStatus.OPTIMAL:
            values[action] = min(values.get(action, np.inf), r.objective)
        elif r.status not in (SolveStatus.INFEASIBLE, SolveStatus.INFEASIBLE_OR_UNBOUNDED):
            return None
    return values


def run(args: "argparse.Namespace | bool" = False) -> None:
    """args from the command line, or smoke (bool) from reproduce.py: smoke runs seed 0 for 5 instants."""
    if isinstance(args, bool):
        args = argparse.Namespace(steps=5 if args else 30, seeds=[0] if args else [0, 1, 2, 3, 4],
                                  benchmark=DEFAULT, smoke=args)
    out = OUT.with_name(OUT.name + "_smoke") if getattr(args, "smoke", False) else OUT
    started, env = datetime.now(timezone.utc), environment(ROOT)
    controllers = {"Gurobi": gurobi(TIME_LIMIT), "B&B": bnb(STRATEGY), "AF+AT": bnb(AFAT)}
    spec = {"name": "closed_loop_benchmarks", "benchmarks": {b: BENCHMARKS[b] for b in args.benchmark},
            "seeds": args.seeds, "steps": args.steps, "time_limit": TIME_LIMIT,
            "strategy": {"name": STRATEGY.name, **asdict(STRATEGY)}, "controllers": list(controllers),
            "timing": "one solve per instant, no repeats"}
    records = []
    for name in args.benchmark:
        for seed in args.seeds:
            benchmark = make_benchmark(name, seed=seed, **BENCHMARKS[name])
            loops = {label: simulate(benchmark, c, args.steps) for label, c in controllers.items()}
            reference = loops["Gurobi"]
            margins, etas = [], []
            u_prev = benchmark.initial_input()
            for k, x in enumerate(reference.states[:-1]):
                problem = benchmark.problem(np.asarray(x), k, u_prev)
                values = action_values(problem)
                if values is None:  # An action-fixed solve reached the time limit.
                    margins.append(None)
                    etas.append(None)
                else:
                    ordered = sorted(values.values())
                    margins.append((ordered[1] - ordered[0]) / abs(ordered[0]) if len(ordered) > 1 else None)
                    etas.append(competing_relaxation_gap(problem, values))
                u_prev = np.asarray(reference.decisions[k].u)
            for label, loop in loops.items():
                records.append({
                    "benchmark": name, "seed": seed, "controller": label,
                    "states": [np.asarray(x).tolist() for x in loop.states],
                    "inputs": loop.inputs.tolist(), "runtimes": loop.runtimes.tolist(),
                    "nodes": [d.nodes for d in loop.decisions], "statuses": [str(d.status) for d in loop.decisions],
                    "margins": margins, "etas": etas,
                })
            totals = {label: loop.runtimes.sum() for label, loop in loops.items()}
            same = all(np.array_equal(loop.inputs, reference.inputs) for loop in loops.values())
            print(f"{name} seed {seed}: totals " + ", ".join(f"{k} {v:.2f} s" for k, v in totals.items())
                  + f"; same inputs {same}; min margin {min((m for m in margins if m is not None), default=None)}; "
                  f"eta > 0 at {sum(e is not None and e > 0 for e in etas)} of {len(etas)} instants "
                  f"({sum(e is None for e in etas)} unresolved)", flush=True)
            save(out, spec, records, started, env)  # After every seed, so an interrupted run keeps its loops.


def assets(smoke: bool = False) -> None:
    out = OUT.with_name(OUT.name + "_smoke") if smoke else OUT
    records, meta = load(out)
    loops = {}
    for r in records:
        loops.setdefault((r["benchmark"], r["seed"]), {})[r["controller"]] = r
    buck = [loop for (b, _), loop in loops.items() if b == "buck"]
    speedups = [sum(l["B&B"]["runtimes"]) / sum(l["AF+AT"]["runtimes"]) for l in buck]
    gurobi = [sum(l["Gurobi"]["runtimes"]) / sum(l["AF+AT"]["runtimes"]) for l in buck]
    early = [b / a for l in buck for a, b in zip(l["AF+AT"]["runtimes"][:EARLY], l["B&B"]["runtimes"][:EARLY])]
    early_max_ms = 1000 * max(t for l in buck for t in l["AF+AT"]["runtimes"][:EARLY])
    print(f"buck: B&B/AF+AT {[round(x, 2) for x in speedups]}, Gurobi/AF+AT {[round(x, 2) for x in gurobi]}")
    macros = {"resLoopBuckSeeds": str(len(buck)),
              "resLoopBuckSpeedupMin": fixed(min(speedups)), "resLoopBuckSpeedupMax": fixed(max(speedups)),
              "resLoopBuckEarlySpeedupMin": f"{min(early):.0f}", "resLoopBuckEarlySpeedupMax": f"{max(early):.0f}",
              "resLoopBuckEarlyMaxMs": f"{early_max_ms:.0f}"}
    write_macros(out / "closed_loop_benchmarks_numbers.tex", macros, SOURCE, meta)
    print(f"Wrote the numbers to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "assets"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--benchmark", nargs="+", default=DEFAULT)
    parser.add_argument("--smoke", action="store_true", help="results/closed_loop_benchmarks_smoke/")
    args = parser.parse_args()
    run(args) if args.command == "run" else assets(args.smoke)


if __name__ == "__main__":
    main()
