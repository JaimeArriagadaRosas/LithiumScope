"""Separated _save_report responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.runtime.preboot import Path, PrebootReport

def _save_report(
    report: PrebootReport,
) -> Path:
    from lithiumscope.runtime.preboot import LOGS_DIR, Path, PrebootReport, asdict, current_error_log_path, current_session_log_path, datetime, json, timezone
    from lithiumscope.core.logging.session import current_session_directory
    from lithiumscope.core.logging.context import unique_id
    from lithiumscope.results.atomic_exports import write_json
    directory = current_session_directory() / "initialization"
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = directory / f"{unique_id('preboot')}.json"
    report.report_path = str(path)
    payload = asdict(report)
    payload["core_ready"] = report.core_ready
    payload["ml_ready"] = report.ml_ready
    payload["imagery_ready"] = report.imagery_ready
    payload["datasets_ready"] = report.datasets_ready
    payload["model_1_ready"] = report.model_1_ready
    payload["model_2_ready"] = report.model_2_ready
    payload["training_ready"] = report.training_ready
    payload["session_log"] = str(
        current_session_log_path()
    )
    payload["error_log"] = str(
        current_error_log_path()
    )
    write_json(path,payload)
    return path
