from __future__ import annotations

import json
from pathlib import Path

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import CONFIG_DIR, RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash, runtime_fingerprint


def build_training_signature(
    model_group: str,
    dataset_path: Path,
    config_names: tuple[str, ...],
) -> str:
    payload = {
        "model_group": model_group,
        "dataset_sha256": file_sha256(dataset_path),
        "git_commit": runtime_fingerprint().get("git_commit"),
        "configs": {
            name: file_sha256(CONFIG_DIR / f"{name}.yaml")
            for name in config_names
        },
    }
    return canonical_json_hash(payload)


def find_compatible_run(model_group: str, signature: str) -> Path | None:
    directory = RESULTS_DIR / model_group / "runs"
    if not directory.exists():
        return None

    for run_dir in sorted(
        (path for path in directory.iterdir() if path.is_dir()),
        reverse=True,
    ):
        run_file = run_dir / "run.json"
        if not run_file.exists():
            continue
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        state = str(payload.get("state", ""))
        summary = payload.get("summary", {})
        if summary.get("training_signature") != signature:
            continue
        if state in {"cancelled", "partial", "running", "completed"}:
            return run_dir
    return None
