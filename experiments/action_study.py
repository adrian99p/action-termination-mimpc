"""Termination and branching study: Table I and the numbers quoted with it.

Every benchmark instance is solved by Gurobi and by B&B with five search strategies,
each without and with action-first branching (AF) and action termination (AT). The paper
reports MF and PC with BFS and DPLG-N. FF-BFS is also run but not reported.

    python -m experiments.action_study run [--benchmark NAME ...]   # solve, save results/action_study/<name>/
    python -m experiments.action_study assets                        # table and numbers from the results

Each benchmark is saved separately, so one benchmark can be rerun alone. assets writes to
results/action_study/. With --smoke, both commands use seed 0 only, in results/action_study_smoke/.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from actionbnb.bnb import BnBConfig
from actionbnb.experiment import Experiment, environment, load, run_experiment, save
from actionbnb.experiment.analysis import select, sgm_nodes, sgm_runtime
from actionbnb.experiment.latex import fixed, header, sig, write, write_macros

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = "experiments/action_study.py"

STRATEGIES = (
    BnBConfig(branching="FF"),
    BnBConfig(branching="MF"),
    BnBConfig(branching="PC"),
    BnBConfig(branching="MF", selection="DPLG", child_order="nearest"),
    BnBConfig(branching="PC", selection="DPLG", child_order="nearest"),
)

EXPERIMENT = Experiment(
    name="action_study",
    benchmarks={"diagnostic": {"N": 16}, "integrator": {"N": 20}, "buck": {"N": 30}, "refrigeration": {"N": 20}},
    configs=tuple(
        replace(strategy, action_first=af, action_termination=at, time_limit=60.0)
        for strategy in STRATEGIES
        for af in (False, True)
        for at in (False, True)
    ),
    seeds=tuple(range(20)),
    repeats=3,
    warmup=True,
)

# Table I: rows per benchmark and rule, columns per termination and branching variant.
LABELS = {"diagnostic": "Diagnostic", "integrator": "Integrator", "buck": "Buck", "refrigeration": "Refrigeration"}
TABLE_RULES = ("MF-DPLG-N", "PC-DPLG-N")
VARIANTS = {"BnB": "", "AT": "-AT", "AF": "-AF", "AFAT": "-AF-AT"}


def experiment(smoke: bool) -> Experiment:
    return replace(EXPERIMENT, name=f"{EXPERIMENT.name}_smoke", seeds=(0,)) if smoke else EXPERIMENT


def run(smoke: bool, benchmarks: tuple[str, ...] = tuple(LABELS)) -> None:
    spec = experiment(smoke)
    for name in benchmarks:
        part = replace(spec, name=f"{spec.name}/{name}", benchmarks={name: spec.benchmarks[name]})
        started, env = datetime.now(timezone.utc), environment(ROOT)  # Code state when the run starts.
        records = run_experiment(part)
        save(RESULTS / part.name, part, records, started, env)
        print(f"Saved {len(records)} runs to {RESULTS / part.name}")


def assets(smoke: bool) -> None:
    out = RESULTS / experiment(smoke).name
    records, metas = [], {}
    for name in LABELS:
        part, metas[name] = load(out / name)
        records += part
    sources = list(metas.values())
    rows = table_rows(records)
    write(out / "action_study_table.tex", header(SOURCE, sources) + latex_table(rows))
    write(out / "action_study_table.md", markdown_table(rows, len(sources[0]["experiment"]["seeds"])))
    write_macros(out / "action_study_numbers.tex", numbers(records, metas), SOURCE, sources)
    print(f"Wrote the table and numbers to {out}")


def table_rows(records: list[dict]) -> list[tuple]:
    """Table I per benchmark and rule: label, rule, SGM runtimes [s] per variant, Gurobi, and certification share."""
    rows = []
    for benchmark, label in LABELS.items():
        gurobi = sgm_runtime(select(records, benchmark=benchmark, solver="Gurobi"))
        for i, rule in enumerate(TABLE_RULES):
            times = [sgm_runtime(select(records, benchmark=benchmark, solver=rule + s)) for s in VARIANTS.values()]
            share = certification_share(records, benchmark, rule)
            rows.append((label if i == 0 else "", rule.split("-")[0], times, gurobi, share))
    return rows


def runtime_cells(times: list[float], bold: str) -> list[str]:
    """Runtimes [ms] of the variants with one decimal; the fastest is wrapped in bold, e.g. '**{}**'."""
    ms = [fixed(1e3 * t) for t in times]
    return [bold.format(m) if t == min(times) else m for t, m in zip(times, ms)]


def latex_table(rows: list[tuple]) -> str:
    """Table I as included by the paper; the fastest variant is bold, and speedup is B&B over AF+AT."""
    lines = []
    for label, rule, times, gurobi, share in rows:
        cells = runtime_cells(times, "\\textbf{{{}}}")
        lines.append(f"{label} & {rule} & {' & '.join(cells)} & {fixed(1e3 * gurobi)}"
                     f" & {fixed(times[0] / times[-1])}$\\times$ & {fixed(share)}\\\\")
    return (
        "\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n"
        "Problem & Rule & B\\&B & AT & AF & AF+AT & Gurobi & Speedup & Cert.\\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def markdown_table(rows: list[tuple], instances: int) -> str:
    """Table I for viewing on GitHub, with the paper's caption."""
    lines = [
        f"SGM runtime [ms] over {instances} instances using DPLG-N. Bold: fastest variant. "
        "Speedup: B&B over AF+AT. Cert.: AF+AT nodes as a percentage of AF nodes.",
        "",
        "| Problem | Rule | B&B | AT | AF | AF+AT | Gurobi | Speedup | Cert. [%] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, rule, times, gurobi, share in rows:
        cells = runtime_cells(times, "**{}**")
        lines.append(f"| {label} | {rule} | {' | '.join(cells)} | {fixed(1e3 * gurobi)}"
                     f" | {fixed(times[0] / times[-1])}× | {fixed(share)} |")
    return "\n".join(lines) + "\n"


def numbers(records: list[dict], metas: dict[str, dict]) -> dict[str, str]:
    """Macros \\res... for the numbers quoted in the text; metas holds the meta data per benchmark."""
    spec = next(iter(metas.values()))["experiment"]  # Seeds and repeats are shared.
    bnb = [r for r in records if r["solver"] != "Gurobi"]
    at_runs = [r for r in bnb if r["action_termination"]]

    # Warn if runs were cut off or contradict a statement in the text.
    if any(r["status"] == "time_limit" for r in bnb):
        print("WARNING: some B&B runs hit the time limit, so their runtimes are only lower bounds.")
    if not all(r["correct"] for r in at_runs):
        print("WARNING: some AT runs did not return an optimal first input, contrary to the text.")
    if any(r["status"] == "action_satisfied" for r in at_runs if not r["action_first"] and r["benchmark"] != "diagnostic"):
        print("WARNING: AT alone certifies early beyond the diagnostic, contrary to the text.")

    def speedup(benchmark: str, rule: str) -> float:
        """SGM runtime of B&B over AF+AT."""
        base, afat = (select(records, benchmark=benchmark, solver=rule + VARIANTS[v]) for v in ("BnB", "AFAT"))
        return sgm_runtime(base) / sgm_runtime(afat)

    control = [b for b in LABELS if b != "diagnostic"]
    macros = {"resInstances": str(len(spec["seeds"])), "resRepeats": str(spec["repeats"]),
              "resDiagnosticMFSpeedup": sig(speedup("diagnostic", "MF-DPLG-N"))}
    for key, rule in (("", "MF-DPLG-N"), ("PC", "PC-DPLG-N")):
        values = [speedup(b, rule) for b in control]
        macros[f"resControlSpeedup{key}Min"], macros[f"resControlSpeedup{key}Max"] = fixed(min(values)), fixed(max(values))
    bfs = [speedup(b, rule) for b in LABELS for rule in ("MF-BFS", "PC-BFS")]
    macros["resBFSSpeedupMin"], macros["resBFSSpeedupMax"] = fixed(min(bfs)), fixed(max(bfs))
    for v in ("BnB", "AFAT"):  # SGM nodes on the control problems with either rule
        nodes = [sgm_nodes(select(records, benchmark=b, solver=rule + VARIANTS[v]))
                 for b in control for rule in TABLE_RULES]
        macros[f"resNodes{v}Min"], macros[f"resNodes{v}Max"] = f"{min(nodes):.0f}", f"{max(nodes):.0f}"
    shares = [certification_share(records, b, rule) for b in LABELS for rule in TABLE_RULES]  # Cert. column
    macros["resCertMin"], macros["resCertMax"] = fixed(min(shares)), fixed(max(shares))
    return macros


def certification_share(records: list[dict], benchmark: str, rule: str) -> float:
    """SGM nodes of AF+AT as a percentage of AF: the share of the AF search needed to certify the first input.

    Both follow the same deterministic search until AF+AT stops at certification.
    """
    af, af_at = (select(records, benchmark=benchmark, solver=rule + VARIANTS[v]) for v in ("AF", "AFAT"))
    return 100 * sgm_nodes(af_at) / sgm_nodes(af)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "assets"))
    parser.add_argument("--benchmark", nargs="+", choices=LABELS, default=list(LABELS), help="run only these")
    parser.add_argument("--smoke", action="store_true", help="seed 0 only, for development")
    args = parser.parse_args()
    if args.command == "run":
        run(args.smoke, args.benchmark)
    else:
        assets(args.smoke)


if __name__ == "__main__":
    main()
