"""Horizon scaling with pseudocost branching: the PC curve of Fig. 3 and its speedup.

The horizon study uses most-fractional branching (MF). This script repeats it for B&B and AF+AT
with pseudocost branching (PC-DPLG-N) on the same instances, with the protocol of horizon.py
run + extend: an untimed warm-up with the 60 s screening limit, then three timed repeats (median
kept) with the 600 s limit, or a single timed run if the warm-up reached the screening limit. The
instance data (V*, action values) come from results/horizon/, so Gurobi is not solved again.

    python -m experiments.horizon_pc run [--benchmark NAME ...]   # saves results/horizon_pc/<name>_N<N>/
    python -m experiments.horizon_pc assets                        # numbers from the saved results

With --smoke, both commands use seed 0 only in results/horizon_pc_smoke/.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from statistics import median

from actionbnb.benchmarks import make_benchmark
from actionbnb.bnb import BnBConfig, solve
from actionbnb.experiment import environment, load, save
from actionbnb.experiment.analysis import select
from actionbnb.experiment.latex import write_macros
from actionbnb.experiment.runner import bnb_record

from experiments import horizon
from experiments.horizon import HORIZONS, LABELS, LIMIT, RESULTS, SCREEN_LIMIT, experiment as mf_experiment, fmt, quantiles

SOURCE = "experiments/horizon_pc.py"
STRATEGY = BnBConfig(branching="PC", selection="DPLG", child_order="nearest", time_limit=LIMIT)
CONFIGS = (STRATEGY, replace(STRATEGY, action_first=True, action_termination=True))
REPEATS = 3


def results(smoke: bool):
    return RESULTS / ("horizon_pc_smoke" if smoke else "horizon_pc")


def timed(problem, config: BnBConfig) -> tuple[list, bool]:
    """Timed results and whether the warm-up reached the screening limit (then one timed run)."""
    warmup = solve(problem, replace(config, time_limit=SCREEN_LIMIT))
    repeats = 1 if warmup.status == "time_limit" else REPEATS
    return [solve(problem, config) for _ in range(repeats)], repeats == 1


def run(smoke: bool, benchmarks: tuple[str, ...] = tuple(LABELS)) -> None:
    for name in benchmarks:
        for N in HORIZONS[name]:
            started, env = datetime.now(timezone.utc), environment(RESULTS.parent)
            instances, _ = load(RESULTS / mf_experiment(name, N, smoke).name)
            records = []
            for r in select(instances, solver="Gurobi"):
                problem = make_benchmark(name, seed=r["seed"], N=N).instance()
                values = {tuple(a): v for a, v in r["action_values"]}
                keep = {k: r[k] for k in ("benchmark", "seed", "N", "v_star", "action_values")}
                for config in CONFIGS:
                    runs, screened = timed(problem, config)
                    records.append(keep | bnb_record(runs, config, values, r["v_star"])
                                   | {"time_limit": LIMIT, "screened_out": screened})
                print(f"{name} N={N} seed {r['seed']}: " + ", ".join(
                    f"{x['solver']} {x['status']} {x['runtime']:.3g} s" for x in records[-2:]), flush=True)
            spec = {"name": f"{results(smoke).name}/{name}_N{N}", "benchmark": name, "N": N, "limit": LIMIT,
                    "screen_limit": SCREEN_LIMIT, "repeats": REPEATS, "configs": [c.name for c in CONFIGS],
                    "instances": f"results/horizon/{name}_N{N}"}
            save(results(smoke) / f"{name}_N{N}", spec, records, started, env)


def assets(smoke: bool) -> None:
    out = results(smoke)
    macros, metas, longest = {}, [], []
    for name in LABELS:
        for N in HORIZONS[name]:
            if not (out / f"{name}_N{N}" / "runs.jsonl").exists():
                continue
            records, meta = load(out / f"{name}_N{N}")
            metas.append(meta)
            bnb, afat = (sorted(select(records, solver=c.name), key=lambda r: r["seed"]) for c in CONFIGS)
            if any(r["status"] == "time_limit" for r in bnb + afat):
                print(f"WARNING: a PC run on {name} N={N} reached the time limit, but the text says none does.")
            ratios = [b["runtime"] / a["runtime"] for b, a in zip(bnb, afat)]
            lower = [b["status"] == "time_limit" for b in bnb]
            if any(a["status"] == "time_limit" for a in afat):
                ratios = [x if a["status"] != "time_limit" else 0.0 for x, a in zip(ratios, afat)]
            q = quantiles(ratios, lower)
            done = (sum(r["status"] != "time_limit" for r in bnb), sum(r["status"] != "time_limit" for r in afat))
            print(f"{name:>13} N={N}: B&B {done[0]}/{len(bnb)} median {median(r['runtime'] for r in bnb):.3g} s, "
                  f"AF+AT {done[1]}/{len(afat)} median {median(r['runtime'] for r in afat):.3g} s, "
                  f"speedup median {fmt(q[1])}, nodes {median(r['nodes'] for r in bnb):.0f} vs "
                  f"{median(r['nodes'] for r in afat):.0f}")
            if N == HORIZONS[name][-1]:
                longest.append(q[1])
    if len(longest) == len(LABELS):
        macros["resHorizonPCSpeedupMin"] = fmt(min(longest, key=lambda v: v[0]))
    if metas:
        write_macros(out / "horizon_pc_numbers.tex", macros, SOURCE, metas[:1])
        horizon.assets(smoke)  # Redraw the horizon figure with the pseudocost curve.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "assets"))
    parser.add_argument("--benchmark", nargs="+", choices=LABELS, default=list(LABELS), help="run only these")
    parser.add_argument("--smoke", action="store_true", help="seed 0 only, for development")
    args = parser.parse_args()
    run(args.smoke, tuple(args.benchmark)) if args.command == "run" else assets(args.smoke)


if __name__ == "__main__":
    main()
