"""Run an experiment: timed, repeated, and verified solves of every instance."""

from collections.abc import Callable
from time import perf_counter

from actionbnb.benchmarks import make_benchmark
from actionbnb.bnb import BnBConfig, BnBResult, solve
from actionbnb.experiment.config import Experiment
from actionbnb.oracle import OracleResult, solve_action_values, solve_with_oracle
from actionbnb.status import SolveStatus

REL_TOL = 1e-6


def run_experiment(experiment: Experiment) -> list[dict]:
    """One record per benchmark, seed, and solver."""
    records: list[dict] = []
    start = perf_counter()
    for name, params in experiment.benchmarks.items():
        for seed in experiment.seeds:
            benchmark = make_benchmark(name, seed=seed, **params)
            problem = benchmark.instance()
            values = solve_action_values(problem)
            v_star = min(values.values())
            instance = {
                "benchmark": name,
                "seed": seed,
                "N": benchmark.N,
                "v_star": v_star,
                "action_values": [[list(action), value] for action, value in values.items()],
            }

            runs = timed(lambda: solve_with_oracle(problem), experiment)
            records.append(instance | gurobi_record(runs, problem, v_star))
            for config in experiment.configs:
                runs = timed(lambda: solve(problem, config), experiment)
                records.append(instance | bnb_record(runs, config, values, v_star))

            print(f"{name} seed {seed:>3}: done, {perf_counter() - start:7.1f} s elapsed", flush=True)

    issues = [r for r in records if not (r["correct"] and r["deterministic"])]
    print(f"{len(records)} runs, {len(issues)} issues.")
    for r in issues:
        print(f"  {r['benchmark']} seed {r['seed']} {r['solver']}: status {r['status']}, "
              f"correct {r['correct']}, deterministic {r['deterministic']}")
    return records


def timed(solve_once: Callable, experiment: Experiment) -> list:
    """Results of the timed repeats, after an optional discarded warm-up."""
    if experiment.warmup:
        solve_once()
    return [solve_once() for _ in range(experiment.repeats)]


def median_run(runs: list):
    return sorted(runs, key=lambda r: r.runtime)[len(runs) // 2]


def close(value: float, reference: float) -> bool:
    return abs(value - reference) <= REL_TOL * max(1.0, abs(reference))


def gurobi_record(runs: list[OracleResult], problem, v_star: float) -> dict:
    r = median_run(runs)
    optimal = r.status == SolveStatus.OPTIMAL
    return {
        "solver": "Gurobi",
        "status": str(r.status),
        "objective": r.objective,
        "action": list(problem.action_encoding.action(r.z)) if optimal else None,
        "runtime": r.runtime,
        "runtimes": [run.runtime for run in runs],
        "nodes": r.nodes_explored,
        "iterations": r.n_simplex,
        "correct": optimal and close(r.objective, v_star),
        "deterministic": len({run.nodes_explored for run in runs}) == 1,
    }


def bnb_record(runs: list[BnBResult], config: BnBConfig, values: dict, v_star: float) -> dict:
    r = median_run(runs)
    if r.status == SolveStatus.OPTIMAL:
        correct = close(r.objective, v_star)
    elif r.status == SolveStatus.ACTION_SATISFIED:
        correct = close(values[r.action], v_star)
    else:
        correct = False
    return {
        "solver": config.name,
        "branching": config.branching,
        "selection": config.selection,
        "child_order": config.child_order,
        "action_first": config.action_first,
        "action_termination": config.action_termination,
        "status": str(r.status),
        "objective": r.objective,
        "action": None if r.action is None else list(r.action),
        "action_value": None if r.action is None else values[r.action],
        "runtime": r.runtime,
        "runtimes": [run.runtime for run in runs],
        "nodes": r.nodes,
        "iterations": r.qp_iterations,
        "qp_solves": r.qp_solves,
        "qp_time": r.qp_time,
        "lower_bound": r.lower_bound,
        "competing_bound": r.competing_bound,
        "first_incumbent_node": r.first_incumbent_node,
        "first_incumbent_time": r.first_incumbent_time,
        "certified_node": r.certified_node,
        "certified_time": r.certified_time,
        "correct": correct,
        "deterministic": len({run.nodes for run in runs}) == 1,
    }
