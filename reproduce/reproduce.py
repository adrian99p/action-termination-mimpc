"""Reproduce every result of the paper: run all experiments, then write their figures, tables, and numbers.

Run from the repository root:

    python -m reproduce.reproduce                         # full runs, about 9 h (7 h for horizon.py)
    python -m reproduce.reproduce --smoke                 # fewer seeds and sampling instants, for a quick check
    python -m reproduce.reproduce --assets-only           # figures, tables, and numbers from the saved results, seconds
    python -m reproduce.reproduce --smoke --assets-only   # the same from the saved smoke results, seconds

Each experiment in experiments/ writes to results/<experiment>/ (results/<experiment>_smoke/ with --smoke).
The paper's figures and Table I are then copied to reproduce/results/ (reproduce/results_smoke/ with --smoke).
The scripts can also be run one at a time; horizon_pc.py uses the results of horizon.py, and
relaxation_gap.py those of action_study.py and closed_loop.py.
"""

import argparse
import shutil
from pathlib import Path

from experiments import (action_study, action_traces, closed_loop, closed_loop_benchmarks, horizon, horizon_pc,
                         relaxation_gap)

EXPERIMENTS = (action_study, action_traces, closed_loop, closed_loop_benchmarks, horizon, horizon_pc, relaxation_gap)
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
PAPER = {  # name in reproduce/results/: (experiment folder in results/, file)
    "fig2_bound_traces.png": ("action_traces", "action_traces.png"),
    "fig3_horizon_scaling.png": ("horizon", "horizon.png"),
    "fig4_closed_loop.png": ("closed_loop", "closed_loop.png"),
    "table1_ablation.md": ("action_study", "action_study_table.md"),
}


def collect(smoke: bool) -> None:
    """Copy the paper's figures and Table I to reproduce/results/."""
    suffix = "_smoke" if smoke else ""
    out = HERE / f"results{suffix}"
    out.mkdir(exist_ok=True)
    for name, (folder, file) in PAPER.items():
        shutil.copy2(RESULTS / f"{folder}{suffix}" / file, out / name)
    print(f"Paper figures and table in {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="fewer seeds and sampling instants, for a quick check")
    parser.add_argument("--assets-only", action="store_true", help="only regenerate outputs from the saved results")
    args = parser.parse_args()
    for experiment in EXPERIMENTS:
        if not args.assets_only:
            experiment.run(args.smoke)
            if hasattr(experiment, "extend"):  # horizon.py: rerun censored runs with the full time limit
                experiment.extend(args.smoke)
        experiment.assets(args.smoke)
    collect(args.smoke)


if __name__ == "__main__":
    main()
