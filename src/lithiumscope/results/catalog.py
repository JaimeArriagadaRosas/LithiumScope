from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.paths import RESULTS_DIR


def load_run_record(run_dir: Path) -> dict:
    path = run_dir / "run.json"
    if not path.exists():
        return {
            "run_id": run_dir.name,
            "model_group": run_dir.parent.parent.name,
            "state": "legacy",
            "summary": {},
            "runtime": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def run_catalog(model_group: str) -> pd.DataFrame:
    directory = RESULTS_DIR / model_group / "runs"
    if not directory.exists():
        return pd.DataFrame()

    records: list[dict] = []
    for run_dir in sorted(
        (path for path in directory.iterdir() if path.is_dir()),
        reverse=True,
    ):
        payload = load_run_record(run_dir)
        summary = payload.get("summary", {})
        runtime = payload.get("runtime", {})
        failed = summary.get("failed_algorithms", [])
        records.append(
            {
                "run_id": payload.get("run_id", run_dir.name),
                "model_group": payload.get("model_group", model_group),
                "state": payload.get("state", "unknown"),
                "winner": summary.get("winner"),
                "primary_metric": summary.get("primary_metric"),
                "primary_metric_value": summary.get("primary_metric_value"),
                "dataset_sha256": summary.get("dataset_sha256"),
                "git_commit": runtime.get("git_commit"),
                "failed_algorithms": ",".join(failed) if failed else "",
                "created_at_utc": payload.get("created_at_utc"),
                "completed_at_utc": payload.get("completed_at_utc"),
                "run_dir": str(run_dir),
                "release_gate_pass": summary.get("release_gate_pass"),
                "release_candidate": (
                    payload.get("state") == "completed"
                    and bool(summary.get("winner"))
                    and bool(summary.get("dataset_sha256"))
                    and bool(runtime.get("git_commit"))
                    and not failed
                    and (
                        model_group != "model_2"
                        or summary.get("release_gate_pass") is True
                    )
                ),
            }
        )
    return pd.DataFrame(records)


def latest_release_candidate(model_group: str) -> dict | None:
    catalog = run_catalog(model_group)
    if catalog.empty:
        return None
    candidates = catalog[catalog["release_candidate"] == True]  # noqa: E712
    if candidates.empty:
        return None
    return candidates.iloc[0].to_dict()
