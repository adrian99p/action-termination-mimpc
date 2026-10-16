"""Horizon scaling: Fig. 3 and the numbers quoted with it.

B&B and AF+AT with MF-DPLG-N, and Gurobi, on the three control problems over a range of
horizons, 20 seeds each, with a common time limit of LIMIT = 600 s for all three solvers.

Protocol. run solves every instance with the protocol of the action study (an untimed warm-up,
then three timed repeats, median kept) and a screening limit of 60 s for B&B. extend then reruns
every B&B run that reached the screening limit once, after a warm-up capped at 5 s, with the full
limit. Gurobi runs without a limit, and assets censors its runtimes above LIMIT. Runs reaching
LIMIT count as unsuccessful: the figure shows quantiles across instances and marks censored ones.

    python -m experiments.horizon run [--benchmark NAME ...]   # solve, save results/horizon/<name>_N<N>/
    python -m experiments.horizon extend                        # rerun censored runs, save results/horizon/extension/
    python -m experiments.horizon assets                        # figure and numbers from the saved results

With --smoke, all commands use seed 0 only in results/horizon_smoke/.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from actionbnb.benchmarks import make_benchmark
from actionbnb.bnb import BnBConfig, solve
from actionbnb.experiment import Experiment, environment, load, run_experiment, save
from actionbnb.experiment.analysis import select
from actionbnb.experiment.latex import fixed, sig, write_macros
from actionbnb.experiment.plotting import BLUE, COLUMN_WIDTH, INK_SECONDARY, ORANGE, save as save_figure, style
from actionbnb.experiment.runner import bnb_record

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = "experiments/horizon.py"

LABELS = {"integrator": "Integrator", "buck": "Buck", "refrigeration": "Refrigeration"}
HORIZONS = {"integrator": (10, 15, 20, 25, 30), "buck": (20, 25, 30, 35, 40), "refrigeration": (10, 15, 20, 25, 30)}
SCREEN_LIMIT = 60.0  # s, limit of run
LIMIT = 600.0  # s, limit of extend and of the reported results, for every solver
WARMUP_LIMIT = 5.0  # s, warm-up of extend
STRATEGY = BnBConfig(branching="MF", selection="DPLG", child_order="nearest", time_limit=SCREEN_LIMIT)
CONFIGS = (STRATEGY, replace(STRATEGY, action_first=True, action_termination=True))
SOLVERS = {"B&B": CONFIGS[0].name, "AF+AT": CONFIGS[1].name, "Gurobi": "Gurobi"}
QUANTILES = (0.25, 0.5, 0.75)


def results(smoke: bool) -> Path:
    return RESULTS / ("horizon_smoke" if smoke else "horizon")


def experiment(name: str, N: int, smoke: bool) -> Experiment:
    return Experiment(name=f"{results(smoke).name}/{name}_N{N}", benchmarks={name: {"N": N}}, configs=CONFIGS,
                      seeds=(0,) if smoke else tuple(range(20)))


def run(smoke: bool, benchmarks: tuple[str, ...] = tuple(LABELS)) -> None:
    for name in benchmarks:
        for N in HORIZONS[name]:
            spec = experiment(name, N, smoke)
            started, env = datetime.now(timezone.utc), environment(ROOT)  # Code state when the run starts.
            records = run_experiment(spec)
            save(RESULTS / spec.name, spec, records, started, env)


def extend(smoke: bool) -> None:
    """Rerun every B&B run of run that reached the screening limit, with the full limit."""
    started, env = datetime.now(timezone.utc), environment(ROOT)
    configs = {c.name: replace(c, time_limit=LIMIT) for c in CONFIGS}
    records = []
    for name in LABELS:
        for N in HORIZONS[name]:
            screened, _ = load(RESULTS / experiment(name, N, smoke).name)
            for r in screened:
                if r["solver"] in configs and r["status"] == "time_limit":
                    problem = make_benchmark(name, seed=r["seed"], N=N).instance()
                    config = configs[r["solver"]]
                    solve(problem, replace(config, time_limit=WARMUP_LIMIT))  # Untimed warm-up.
                    result = solve(problem, config)
                    values = {tuple(a): v for a, v in r["action_values"]}
                    keep = {k: r[k] for k in ("benchmark", "seed", "N", "v_star", "action_values")}
                    records.append(keep | bnb_record([result], config, values, r["v_star"]) | {"time_limit": LIMIT})
                    print(f"{name} N={N} seed {r['seed']} {r['solver']}: {result.status}, "
                          f"{result.runtime:.1f} s, {result.nodes} nodes", flush=True)
    spec = {"name": f"{results(smoke).name}/extension", "limit": LIMIT, "warmup_limit": WARMUP_LIMIT,
            "screen_limit": SCREEN_LIMIT, "repeats": 1, "configs": list(configs)}
    save(results(smoke) / "extension", spec, records, started, env)


def merged(smoke: bool) -> tuple[dict, list[dict]]:
    """Records per (benchmark, N): screened runs with censored runs replaced by the extension, Gurobi censored."""
    out = results(smoke)
    extension, ext_meta = load(out / "extension") if (out / "extension" / "runs.jsonl").exists() else ([], None)
    replacement = {(r["benchmark"], r["N"], r["seed"], r["solver"]): r for r in extension}
    cells, metas = {}, []
    for name in LABELS:
        for N in HORIZONS[name]:
            records, meta = load(RESULTS / experiment(name, N, smoke).name)
            metas.append(meta)
            cell = []
            for r in records:
                r = replacement.get((name, N, r["seed"], r["solver"]), r)
                if r["status"] == "time_limit" and r.get("time_limit") != LIMIT:
                    raise RuntimeError(f"{name} N={N} seed {r['seed']} {r['solver']} was not extended to {LIMIT} s.")
                if r["runtime"] >= LIMIT:  # Gurobi without a limit, or a run that reached LIMIT
                    r = r | {"status": "time_limit", "runtime": LIMIT, "correct": False}
                cell.append(r)
            cells[name, N] = cell
    return cells, metas + ([ext_meta] if ext_meta else [])


def quantiles(values: list[float], censored: list[bool]) -> list[tuple[float, bool]]:
    """Quantiles of values (linear interpolation) and whether each depends on a censored value."""
    order = np.argsort(values, kind="stable")
    v, c = np.asarray(values, float)[order], np.asarray(censored)[order]
    out = []
    for q in QUANTILES:
        position = q * (len(v) - 1)
        lo, hi = int(position), ceil(position)
        out.append((float(v[lo] + (position - lo) * (v[hi] - v[lo])), bool(c[hi])))
    return out


def cell_stats(cell: list[dict]) -> dict:
    """Completion, runtime quantiles, median nodes, and paired median runtime ratios per solver."""
    runs = {label: sorted(select(cell, solver=s), key=lambda r: r["seed"]) for label, s in SOLVERS.items()}
    stats = {}
    for label, rs in runs.items():
        censored = [r["status"] == "time_limit" for r in rs]
        stats[label] = {"done": len(rs) - sum(censored), "n": len(rs),
                        "runtime": quantiles([r["runtime"] for r in rs], censored),
                        "nodes": float(np.median([r["nodes"] for r in rs]))}
    for other in ("B&B", "Gurobi"):  # Ratio other / AF+AT per instance; a censored numerator is a lower bound.
        ratios = [o["runtime"] / a["runtime"] for o, a in zip(runs[other], runs["AF+AT"])]
        lower = [o["status"] == "time_limit" for o in runs[other]]
        if any(a["status"] == "time_limit" for a in runs["AF+AT"]):
            ratios = [x if a["status"] != "time_limit" else 0.0 for x, a in zip(ratios, runs["AF+AT"])]  # Unknown: worst
        stats[f"{other}/AF+AT"] = quantiles(ratios, lower)
    return stats


def pc_stats(stats: dict, smoke: bool) -> None:
    """Add B&B with pseudocost branching (results/horizon_pc/, from horizon_pc.py) to stats, where run."""
    folder = RESULTS / ("horizon_pc_smoke" if smoke else "horizon_pc")
    for (name, N), s in stats.items():
        if not (folder / f"{name}_N{N}" / "runs.jsonl").exists():
            continue
        records, _ = load(folder / f"{name}_N{N}")
        rs = select(records, solver="PC-DPLG-N")
        s["B&B (PC)"] = {"runtime": quantiles([r["runtime"] for r in rs], [r["status"] == "time_limit" for r in rs])}


def fmt(value: tuple[float, bool], decimals: int | None = None) -> str:
    """A ratio quantile, with a lower-bound sign if it depends on a censored run."""
    return ("$\\geq$" if value[1] else "") + (sig(value[0]) if decimals is None else fixed(value[0], decimals))


def assets(smoke: bool) -> None:
    out = results(smoke)
    cells, metas = merged(smoke)
    stats = {key: cell_stats(cell) for key, cell in cells.items()}
    limited = {label: sum(s[label]["n"] - s[label]["done"] for s in stats.values()) for label in SOLVERS}
    if limited["AF+AT"]:
        print("WARNING: AF+AT reached the time limit, but the text says it never does.")
    longest = [stats[name, HORIZONS[name][-1]] for name in LABELS]
    macros = {"resHorizonLimit": f"{LIMIT:g}", "resHorizonRuns": str(sum(s["B&B"]["n"] for s in stats.values())),
              "resHorizonBnBLimited": str(limited["B&B"]), "resHorizonGurobiLimited": str(limited["Gurobi"]),
              "resHorizonAFATNodesMin": f"{min(s['AF+AT']['nodes'] for s in longest):.0f}",
              "resHorizonAFATNodesMax": f"{max(s['AF+AT']['nodes'] for s in longest):.0f}",
              "resHorizonGurobiRatioMin": fmt(min((s["Gurobi/AF+AT"][1] for s in longest), key=lambda v: v[0]), 0),
              "resHorizonGurobiRatioMax": fmt(max((s["Gurobi/AF+AT"][1] for s in longest), key=lambda v: v[0]), 0)}
    for (name, N), s in stats.items():
        print(f"{name:>13} N={N}: " + "  ".join(
            f"{k} {s[k]['done']}/{s[k]['n']} median {s[k]['runtime'][1][0]:.3g} s" for k in SOLVERS)
            + f"  speedup median {fmt(s['B&B/AF+AT'][1])}")
    write_macros(out / "horizon_numbers.tex", macros, SOURCE, metas[:1] + metas[-1:])
    pc_stats(stats, smoke)
    horizon_figure(stats, out / "horizon.pdf")
    print(f"Wrote the figure and numbers to {out}")


SERIES = {  # Stats key: legend label, color, line style, marker, marker face
    "B&B": ("B&B, MF", BLUE, "-", "o", "white"),
    "B&B (PC)": ("B&B, PC", BLUE, (0, (3, 1.5)), "s", "white"),
    "AF+AT": ("AF+AT, MF", ORANGE, "-", "o", ORANGE),
    "Gurobi": ("Gurobi", INK_SECONDARY, (0, (1, 1.2)), "x", INK_SECONDARY),
}


def horizon_figure(stats: dict, path: Path) -> None:
    """Horizon figure (Fig. 3): median runtime across instances against N, with the interquartile range shaded."""
    style()
    fig, axes = plt.subplots(1, len(LABELS), figsize=(COLUMN_WIDTH, 1.55), sharey=True)
    for ax, name in zip(axes, LABELS):
        Ns = np.array(HORIZONS[name])
        for label, (_, color, dash, marker, face) in SERIES.items():
            if any(label not in stats[name, N] for N in Ns):
                continue
            q = np.array([[v for v, _ in stats[name, N][label]["runtime"]] for N in Ns]) * 1e3
            censored = np.array([stats[name, N][label]["runtime"][1][1] for N in Ns])
            ax.fill_between(Ns, q[:, 0], q[:, 2], color=color, alpha=0.12 if label in ("Gurobi", "B&B (PC)") else 0.18,
                            linewidth=0, zorder=1)
            ax.plot(Ns, q[:, 1], color=color, linestyle=dash, linewidth=1.0, zorder=2)
            ax.plot(Ns[~censored], q[~censored, 1], linestyle="none", marker=marker, markersize=3,
                    color=color, markerfacecolor=face, zorder=3)
            if censored.any():  # A censored median is a lower bound: an upward triangle.
                ax.plot(Ns[censored], q[censored, 1], linestyle="none", marker="^", markersize=3.5,
                        color=color, markerfacecolor="white", zorder=3)
        ax.axhline(LIMIT * 1e3, color=INK_SECONDARY, linewidth=0.5, linestyle=(0, (4, 2)), zorder=0)
        ax.set_yscale("log")
        ax.set_xticks(Ns[::2])
        ax.set_title(LABELS[name], loc="left", pad=3)
        ax.set_xlabel("Horizon $N$")
    axes[0].set_ylabel("Runtime [ms]")
    axes[0].set_ylim(top=LIMIT * 1e3 * 2)
    axes[1].annotate(f"{LIMIT:g} s limit", (axes[1].get_xlim()[0], LIMIT * 1e3), xytext=(2, -1.5),
                     textcoords="offset points", fontsize=6, color=INK_SECONDARY, va="top")
    shown = [key for key in SERIES if all(key in s for s in stats.values())]
    handles = [Line2D([], [], color=c, linestyle=d, marker=m, markersize=3, markerfacecolor=f, label=label)
               for key, (label, c, d, m, f) in SERIES.items() if key in shown]
    handles.append(Patch(facecolor=INK_SECONDARY, alpha=0.25, linewidth=0, label="25–75%"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.04),
               handlelength=1.6, columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(w_pad=0.4, rect=(0, 0, 1, 0.9))
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "extend", "assets"))
    parser.add_argument("--benchmark", nargs="+", choices=LABELS, default=list(LABELS), help="run only these")
    parser.add_argument("--smoke", action="store_true", help="seed 0 only, for development")
    args = parser.parse_args()
    if args.command == "run":
        run(args.smoke, tuple(args.benchmark))
    elif args.command == "extend":
        extend(args.smoke)
    else:
        assets(args.smoke)


if __name__ == "__main__":
    main()
