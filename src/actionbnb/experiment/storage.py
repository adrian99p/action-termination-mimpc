"""Experiment files on disk: raw results (runs.jsonl, one run per line) and meta.json."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import gurobipy as gp

from actionbnb.experiment.config import Experiment


def save(directory: Path, experiment: Experiment | dict, records: list[dict], started: datetime,
         env: dict | None = None) -> None:
    """experiment is an Experiment or a description dict with at least a name.

    env is environment() taken when the run started; taken now if omitted.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runs.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    meta = {
        "experiment": experiment if isinstance(experiment, dict) else experiment.to_dict(),
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": env or environment(directory),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load(directory: Path) -> tuple[list[dict], dict]:
    with (directory / "runs.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    return records, meta


def environment(directory: Path) -> dict:
    """Code revision, software versions, and machine."""
    return {
        "git_commit": _git(directory, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(directory, "status", "--porcelain", "--ignore-submodules=all",
                                "--", ":/", ":!/results")),
        "python": sys.version.split()[0],
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "gurobipy": version("gurobipy"),
        "gurobi": ".".join(map(str, gp.gurobi.version())),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _git(directory: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()
