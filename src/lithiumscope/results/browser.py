from __future__ import annotations

import os
from pathlib import Path
import webbrowser

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.results.dashboard import open_dashboard


def list_runs(model_group: str) -> list[Path]:
    directory = RESULTS_DIR / model_group / "runs"
    if not directory.exists():
        return []
    return sorted((path for path in directory.iterdir() if path.is_dir()), reverse=True)


def latest_run(model_group: str) -> Path | None:
    runs = list_runs(model_group)
    return runs[0] if runs else None


def preview_latest(model_group: str) -> Path | None:
    run = latest_run(model_group)
    return open_dashboard(run) if run else None


def open_latest_excel(model_group: str) -> Path | None:
    run = latest_run(model_group)
    if run is None:
        return None
    candidates = sorted((run / "exports").glob("*.xlsx"))
    if not candidates:
        return None
    path = candidates[0]
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.resolve().as_uri())
    return path


def open_results_folder() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(RESULTS_DIR)  # type: ignore[attr-defined]
    else:
        webbrowser.open(RESULTS_DIR.resolve().as_uri())
