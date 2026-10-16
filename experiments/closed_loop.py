"""Closed-loop study: Fig. 4.

The integrator (seed 2) runs under three controllers that solve
every sampling instant from scratch: Gurobi's full MIQP solve, conventional B&B, and
B&B with action-first branching and action termination (AF+AT), both MF-DPLG-N.

    python -m experiments.closed_loop run       # simulate, save results/closed_loop/
    python -m experiments.closed_loop assets    # figure from the results

assets writes to results/closed_loop/. With --smoke, both commands use 5 sampling instants in
results/closed_loop_smoke/.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MultipleLocator

from actionbnb.benchmarks import make_benchmark
from actionbnb.bnb import BnBConfig
from actionbnb.experiment import environment, load, save
from actionbnb.experiment.plotting import BLUE, COLUMN_WIDTH, INK, INK_SECONDARY, ORANGE, save as save_figure, style
from actionbnb.mpc import bnb, gurobi, simulate

ROOT = Path(__file__).resolve().parents[1]

BENCHMARK, PARAMS = "integrator", {"seed": 2, "N": 20}
STEPS, SMOKE_STEPS = 20, 5
TIME_LIMIT = 60.0
STRATEGY = BnBConfig(branching="MF", selection="DPLG", child_order="nearest", time_limit=TIME_LIMIT)
CONTROLLERS = {
    "Gurobi": gurobi(TIME_LIMIT),
    "B&B": bnb(STRATEGY),
    "AF+AT": bnb(replace(STRATEGY, action_first=True, action_termination=True)),
}
STYLE = {  # Per controller: color, line style, marker, and marker face; drawn in this order.
    "B&B": (BLUE, "-", "o", "white"),
    "AF+AT": (ORANGE, (0, (3.5, 1.5)), "o", ORANGE),
    "Gurobi": (INK_SECONDARY, (0, (1, 1.2)), "x", "none"),
}


def results(smoke: bool) -> Path:
    return ROOT / "results" / ("closed_loop_smoke" if smoke else "closed_loop")


def run(smoke: bool) -> None:
    started, env = datetime.now(timezone.utc), environment(ROOT)  # Code state when the run starts.
    steps = SMOKE_STEPS if smoke else STEPS
    benchmark = make_benchmark(BENCHMARK, **PARAMS)
    records = []
    for label, controller in CONTROLLERS.items():
        loop = simulate(benchmark, controller, steps)
        records.append({
            "controller": label,
            "states": [x.tolist() for x in loop.states],
            "inputs": loop.inputs.tolist(),
            "runtimes": loop.runtimes.tolist(),
            "nodes": [d.nodes for d in loop.decisions],
            "statuses": [str(d.status) for d in loop.decisions],
        })
        print(f"{label}: total {loop.runtimes.sum():.2f} s, max {loop.runtimes.max():.2f} s")
    spec = {"name": results(smoke).name, "benchmark": BENCHMARK, **PARAMS, "steps": steps,
            "time_limit": TIME_LIMIT, "strategy": {"name": STRATEGY.name, **asdict(STRATEGY)},
            "controllers": list(CONTROLLERS)}
    save(results(smoke), spec, records, started, env)
    print(f"Saved to {results(smoke)}")


def assets(smoke: bool) -> None:
    out = results(smoke)
    records, _ = load(out)
    loops = {r["controller"]: r for r in records}
    if any(r["inputs"] != loops["Gurobi"]["inputs"] for r in records):
        print("WARNING: the controllers apply different inputs, but the text says they coincide.")
    closed_loop_figure(loops, out / "closed_loop.pdf")
    print(f"Wrote the figure to {out}")


def closed_loop_figure(loops: dict[str, dict], path: Path) -> None:
    """Closed-loop figure: states, inputs, and solve time per instant for each controller, overlaid."""
    style()
    fig, (ax_x, ax_u, ax_t) = plt.subplots(3, 1, sharex=True, figsize=(COLUMN_WIDTH, 2.0),
                                           gridspec_kw={"height_ratios": [1.1, 0.65, 1.3]})
    for label in STYLE:
        color, dash, marker, face = STYLE[label]
        r = loops[label]
        states, inputs, runtimes = np.array(r["states"]), np.array(r["inputs"])[:, 0], np.array(r["runtimes"])
        instants, steps = np.arange(len(states)), np.arange(len(inputs))
        line = {"color": color, "linestyle": dash, "linewidth": 1.1}
        for column in range(states.shape[1]):
            ax_x.plot(instants, states[:, column], **line)
        ax_u.step(np.r_[steps, steps[-1] + 1], np.r_[inputs, inputs[-1]], where="post", **line)
        ax_t.plot(steps, 1e3 * runtimes, marker=marker, markersize=3, label=label, markerfacecolor=face, **line)

    for column, name, extreme in ((0, "p", np.argmin), (1, "v", np.argmax)):  # Direct labels at each extreme.
        k = int(extreme(states[:, column]))
        above = extreme is np.argmax
        ax_x.annotate(f"${name}$", (instants[k], states[k, column]), xytext=(0, 3 if above else -3),
                      textcoords="offset points", ha="center", va="bottom" if above else "top", fontsize=7, color=INK)
    ax_x.axhline(0.0, color=INK_SECONDARY, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    ax_x.set_ylabel("State")
    ax_x.set_ylim(states.min() - 2.2, states.max() + 1.6)
    ax_u.set_yticks([-1, 0, 1])
    ax_u.set_ylim(-1.4, 1.4)
    ax_u.set_ylabel("$u$")
    ax_t.set_yscale("log")
    ax_t.set_ylim(1, 1e5)
    ax_t.yaxis.set_major_locator(LogLocator(base=100, numticks=4))
    ax_t.xaxis.set_major_locator(MultipleLocator(5))
    ax_t.set_ylabel("Time [ms]")

    ax_t.set_xlim(-0.5, len(instants) - 0.5)
    ax_t.set_xlabel("Sampling instant $k$")

    fig.legend(*ax_t.get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.4)
    fig.align_ylabels()
    fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 0.93))
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "assets"))
    parser.add_argument("--smoke", action="store_true", help=f"{SMOKE_STEPS} sampling instants, for development")
    args = parser.parse_args()
    if args.command == "run":
        run(args.smoke)
    else:
        assets(args.smoke)


if __name__ == "__main__":
    main()
