"""Relaxation separation: when action termination certifies early in the closed loop.

For an instance with optimal first input a*, let Lambda(a) be the QP relaxation bound with the
encoding fixed to a pattern of a, and define the competing relaxation gap

    eta = (V* - min_{a != a*} Lambda(a)) / |V*|.

If eta <= 0, the relaxation separates the first inputs. Computed for the open-loop instances of
the action study and for every sampling instant of the closed loop, next to the observed
certification effort; run it after action_study.py and closed_loop.py. The text quotes the
closed-loop numbers.

    python -m experiments.relaxation_gap run      # solve the relaxations, save results/relaxation_gap/
    python -m experiments.relaxation_gap assets   # numbers from the saved results

With --smoke, both commands use the smoke results in results/relaxation_gap_smoke/.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from actionbnb.benchmarks import make_benchmark
from actionbnb.experiment import environment, load, save
from actionbnb.experiment.latex import fixed, write_macros
from actionbnb.oracle import solve_action_values, solve_with_oracle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = "experiments/relaxation_gap.py"
BENCHMARKS = {"diagnostic": {"N": 16}, "integrator": {"N": 20}, "buck": {"N": 30}, "refrigeration": {"N": 20}}


def results(smoke: bool) -> Path:
    return RESULTS / ("relaxation_gap_smoke" if smoke else "relaxation_gap")


def source(name: str, smoke: bool) -> Path:
    return RESULTS / (f"{name}_smoke" if smoke else name)


def competing_relaxation_gap(problem, values: dict) -> float:
    """eta, relative to V*."""
    v_star = min(values.values())
    best = min(values, key=values.get)
    relax = replace(problem, binary_idx=np.array([], dtype=int))
    idx = list(problem.action_encoding.indices)
    bounds = []
    for pattern, action in problem.action_encoding.patterns.items():
        if action != best:
            lb, ub = relax.lb.copy(), relax.ub.copy()
            lb[idx] = ub[idx] = pattern
            result = solve_with_oracle(replace(relax, lb=lb, ub=ub))
            bounds.append(np.inf if result.objective is None else result.objective)
    return (v_star - min(bounds)) / abs(v_star)


def run(smoke: bool) -> None:
    started, env = datetime.now(timezone.utc), environment(ROOT)  # Code state when the run starts.
    records = []
    for name, params in BENCHMARKS.items():
        runs, _ = load(source("action_study", smoke) / name)
        by_seed = {r["seed"]: r for r in runs if r["solver"] == "MF-DPLG-N-AF-AT"}
        for seed, r in sorted(by_seed.items()):
            values = {tuple(a): v for a, v in r["action_values"]}
            problem = make_benchmark(name, seed=seed, **params).instance()
            records.append({"kind": "open_loop", "benchmark": name, "seed": seed,
                            "eta": competing_relaxation_gap(problem, values)})
    loops, meta = load(source("closed_loop", smoke))
    loops = {r["controller"]: r for r in loops}
    spec = meta["experiment"]
    benchmark = make_benchmark(spec["benchmark"], seed=spec["seed"], N=spec["N"])
    for k, x in enumerate(loops["Gurobi"]["states"][:-1]):
        problem = benchmark.problem(np.array(x), k)
        values = solve_action_values(problem)
        ordered = sorted(values.values())
        records.append({"kind": "closed_loop", "k": k, "margin": (ordered[1] - ordered[0]) / ordered[0],
                        "eta": competing_relaxation_gap(problem, values), "afat_runtime": loops["AF+AT"]["runtimes"][k]})
    save(results(smoke), {"name": results(smoke).name, "benchmarks": BENCHMARKS}, records, started, env)
    print(f"Saved {len(records)} records to {results(smoke)}")


def assets(smoke: bool) -> None:
    records, meta = load(results(smoke))
    loop = [r for r in records if r["kind"] == "closed_loop"]
    fast, slow = [r for r in loop if r["eta"] <= 0], [r for r in loop if r["eta"] > 0]
    macros = {"resEtaFastMaxMs": fixed(1e3 * max(r["afat_runtime"] for r in fast))}
    if slow:
        macros |= {"resEtaSlowMin": fixed(100 * min(r["eta"] for r in slow)),
                   "resEtaSlowMax": fixed(100 * max(r["eta"] for r in slow)),
                   "resMarginSlowMin": fixed(100 * min(r["margin"] for r in slow))}
    out = results(smoke)
    write_macros(out / "relaxation_gap_numbers.tex", macros, SOURCE, meta)
    print(f"Wrote the numbers to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "assets"))
    parser.add_argument("--smoke", action="store_true", help="use the smoke results, for development")
    args = parser.parse_args()
    if args.command == "run":
        run(args.smoke)
    else:
        assets(args.smoke)


if __name__ == "__main__":
    main()
