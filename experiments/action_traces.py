"""Bound traces: Fig. 2.

Every seed of each benchmark is solved with PC-DPLG-N and action-first branching,
tracking actions but without action termination, so the search continues past
certification to full optimality. The certification share is the certification node
over the nodes to optimality. The figure shows the seed with the median share per
benchmark, so the illustrated search is representative rather than selected by eye.

    python -m experiments.action_traces run      # solve, save results/action_traces/
    python -m experiments.action_traces assets   # figure from the results

assets writes to results/action_traces/. With --smoke, both commands use seeds 0-2 in
results/action_traces_smoke/.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from actionbnb.benchmarks import make_benchmark
from actionbnb.bnb import BnBConfig, solve
from actionbnb.experiment import environment, load, save
from actionbnb.experiment.plotting import (BLUE, INK, INK_SECONDARY, LABEL_BOX, ORANGE, TEXT_WIDTH, save as save_figure,
                                           style)
from actionbnb.oracle import solve_with_oracle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "action_traces"

CONFIG = BnBConfig(branching="PC", selection="DPLG", child_order="nearest", action_first=True, trace=True)
BENCHMARKS = {"diagnostic": {"N": 16}, "integrator": {"N": 20}, "buck": {"N": 30}, "refrigeration": {"N": 20}}
SEEDS, SMOKE_SEEDS = tuple(range(20)), (0, 1, 2)
LABELS = {"diagnostic": "Diagnostic", "integrator": "Integrator", "buck": "Buck", "refrigeration": "Refrigeration"}
LINTHRESH = {"diagnostic": 100.0, "integrator": 1.0, "buck": 10.0, "refrigeration": 1.0}  # Linear gap range [%]
LINSCALE = {"diagnostic": 2.4}  # Height of the linear range in decades; 1.2 otherwise
BOTTOM = {"diagnostic": -5e4}  # Lower gap limit [%], with room for the optimal-stop label; one decade below otherwise
OPTIMAL_LABEL = {"diagnostic": -3e3}  # Height of the optimal-stop label [%]; the free bottom decade otherwise
LINEAR_NODES = 200  # Searches this short get a linear node axis.
DASHED = (0, (3.5, 1.5))


def results(smoke: bool) -> Path:
    return RESULTS.with_name(RESULTS.name + "_smoke") if smoke else RESULTS


def run(smoke: bool) -> None:
    """Trace every seed; keep the full trace of the median-share seed and a summary of all seeds."""
    started, env = datetime.now(timezone.utc), environment(ROOT)  # Code state when the run starts.
    records = []
    for name, params in BENCHMARKS.items():
        seeds = SMOKE_SEEDS if smoke else SEEDS
        solved = {seed: solve(make_benchmark(name, seed=seed, **params).instance(), CONFIG) for seed in seeds}
        shares = {seed: r.certified_node / r.nodes for seed, r in solved.items()}
        median = float(np.median(list(shares.values())))
        seed = min(seeds, key=lambda s: (abs(shares[s] - median), s))
        result = solved[seed]

        problem = make_benchmark(name, seed=seed, **params).instance()
        v_star = solve_with_oracle(problem).objective
        assert abs(result.objective - v_star) <= 1e-6 * max(1.0, abs(v_star)), f"{name}: not optimal"
        records.append({
            "benchmark": name,
            "seed": seed,
            **params,
            "solver": CONFIG.name,
            "status": str(result.status),
            "objective": result.objective,
            "v_star": v_star,
            "nodes": result.nodes,
            "first_incumbent_node": result.first_incumbent_node,
            "certified_node": result.certified_node,
            "median_share": median,
            "seeds": [{"seed": s, "certified_node": r.certified_node, "nodes": r.nodes} for s, r in solved.items()],
            "trace": asdict(result.trace),
        })
        print(f"{name}: median share {100 * median:.1f}%, seed {seed} certified at node "
              f"{result.certified_node} of {result.nodes}")
    spec = {"name": "action_traces", "config": {"name": CONFIG.name, **asdict(CONFIG)},
            "benchmarks": BENCHMARKS, "seeds": list(seeds), "selection": "median certification share"}
    save(results(smoke), spec, records, started, env)
    print(f"Saved to {results(smoke)}")


def assets(smoke: bool) -> None:
    out = results(smoke)
    records, _ = load(out)
    trace_figure(records, out / "action_traces.pdf")
    print(f"Wrote the figure to {out}")


def trace_figure(records: list[dict], path: Path) -> None:
    """Trace figure: bounds U, L, and L_{!=a} as gaps to the optimum over the nodes of one search per benchmark.

    The action stop is where L_{!=a} reaches U; the search continues to optimality, where L reaches U.
    """
    style()
    fig, axes = plt.subplots(1, len(records), figsize=(TEXT_WIDTH, 1.8))
    for ax, r in zip(axes, records):
        trace = r["trace"]
        nodes = np.array(trace["nodes"], dtype=float)
        v_star, certified, optimal = r["v_star"], r["certified_node"], r["nodes"]
        shown = nodes >= 1
        nodes = nodes[shown]
        upper, lower, competing = (
            100 * (np.array(trace[k], dtype=float)[shown] - v_star) / abs(v_star) for k in ("upper", "lower", "competing")
        )
        axis = GapAxis(LINTHRESH[r["benchmark"]], np.concatenate([upper, lower, competing]),
                       LINSCALE.get(r["benchmark"], 1.2), BOTTOM.get(r["benchmark"]))
        for values, color, dash in ((lower, BLUE, "-"), (competing, ORANGE, DASHED), (upper, INK, "-")):
            ax.step(nodes, axis.clip(values), where="post", color=color, linewidth=1.1, linestyle=dash)

        if optimal <= LINEAR_NODES:
            ax.set_xlim(0, optimal * 1.03)
        else:
            ax.set_xscale("log")
            ax.set_xlim(0.8, optimal * 1.6)
        axis.apply(ax)
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
        ax.axvspan(certified, optimal, color=ORANGE, alpha=0.07, linewidth=0)
        ax.axvline(certified, color=ORANGE, linewidth=0.9, linestyle=":", zorder=1.2)
        ax.axvline(optimal, color=INK_SECONDARY, linewidth=0.9, linestyle=":", zorder=1.2)
        right = (np.log(certified / 0.8) / np.log(optimal * 2) if optimal > LINEAR_NODES else certified / optimal) < 0.6
        # The top label may cover vertical jumps; the bottom one covers only the dotted stop lines and stays
        # below the bound curves.
        stop_label(ax, f"action stop: {certified:,} nodes", certified, axis.top_band, 1 if right else -1, zorder=6)
        stop_label(ax, f"optimal: {optimal:,} nodes", optimal, OPTIMAL_LABEL.get(r["benchmark"], axis.bottom_band), -1,
                   zorder=1.5)
        ax.set_xlabel("Nodes explored")
        ax.set_title(f"{LABELS[r['benchmark']]}, seed {r['seed']}", loc="left", pad=6)
    axes[0].set_ylabel("Gap to optimum")

    handles = [
        Line2D([], [], color=INK, linewidth=1.1, label=r"Incumbent $U$"),
        Line2D([], [], color=BLUE, linewidth=1.1, label=r"Lower bound $L$"),
        Line2D([], [], color=ORANGE, linewidth=1.1, linestyle=DASHED, label=r"Action bound $L_{\neq}$"),
        Patch(facecolor=ORANGE, alpha=0.14, linewidth=0, label="Search saved by action termination"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=2.0,
               columnspacing=1.6)
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.88))

    save_figure(fig, path)


def stop_label(ax: plt.Axes, text: str, x: float, y: float, side: int, zorder: float) -> None:
    """Label a stop line at height y (data) on the given side of it."""
    ax.annotate(text.replace(",", " "), (x, y), xytext=(3 * side, 0), textcoords="offset points",
                ha="left" if side > 0 else "right", va="center", fontsize=6.5, color=INK, bbox=LABEL_BOX, zorder=zorder)


class GapAxis:
    """Symlog gap axis [%], with a row for +inf (and -inf if needed) beyond a break in the axis."""

    def __init__(self, threshold: float, values: np.ndarray, linscale: float = 1.2, bottom: float | None = None):
        finite = values[np.isfinite(values)]
        self.threshold, self.linscale, self.bottom = threshold, linscale, bottom
        self.hi = 10 ** np.ceil(np.log10(max(finite.max(), threshold)))
        self.lo = -(10 ** np.ceil(np.log10(max(-finite.min(), threshold))))
        # A free decade above and below the finite values holds the stop labels.
        self.top_band, self.bottom_band = self.hi * 5, self.lo * 5
        self.plus_inf = self.hi * 100
        self.minus_inf = self.lo * 100 if np.isneginf(values).any() else None
        self.breaks = [self.hi * 10] + ([] if self.minus_inf is None else [self.lo * 10])

    def clip(self, values: np.ndarray) -> np.ndarray:
        return np.where(np.isposinf(values), self.plus_inf, np.where(np.isneginf(values), self.minus_inf or 0, values))

    def apply(self, ax: plt.Axes) -> None:
        ax.set_yscale("symlog", linthresh=self.threshold, linscale=self.linscale)
        bottom = self.lo * 10 if self.minus_inf is None else self.minus_inf * 2
        ax.set_ylim(bottom if self.bottom is None else self.bottom, self.plus_inf * 2)
        extremes = [self.plus_inf] + ([] if self.minus_inf is None else [self.minus_inf])
        ax.yaxis.set_major_locator(FixedLocator(sorted(self.ticks() + extremes)))
        ax.yaxis.set_major_formatter(FuncFormatter(self.label))
        ax.yaxis.set_minor_locator(NullLocator())
        for y in self.breaks:  # Gap across the plot and a break mark on the axis.
            ax.axhline(y, color="white", linewidth=3.0, zorder=2.5)
            ax.plot([-0.025, 0.025], [y / 1.6, y * 1.6], transform=ax.get_yaxis_transform(), color=INK_SECONDARY,
                     linewidth=0.6, clip_on=False, zorder=4)

    def ticks(self) -> list[float]:
        """0 and at most two decades per side, from the limit toward the linear threshold."""
        def decades(limit: float) -> list[float]:
            exponents = np.arange(np.log10(self.threshold), np.log10(limit) + 0.5)
            step = int(np.ceil(len(exponents) / 2))
            return [10.0 ** e for e in exponents[::-1][::step]]
        return sorted([-d for d in decades(-self.lo)] + [0.0] + decades(self.hi))

    def label(self, value: float, _) -> str:
        if value == self.plus_inf:
            return r"$+\infty$"
        if value == self.minus_inf:
            return r"$-\infty$"
        if value == 0:
            return "0%"
        exponent = round(np.log10(abs(value)))
        sign = "" if value > 0 else "-"
        body = f"{abs(value):g}" if exponent <= 3 else f"10^{{{exponent}}}"
        return f"${sign}{body}\\%$"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "assets"))
    parser.add_argument("--smoke", action="store_true", help="seeds 0-2, for development")
    args = parser.parse_args()
    if args.command == "run":
        run(args.smoke)
    else:
        assets(args.smoke)


if __name__ == "__main__":
    main()
