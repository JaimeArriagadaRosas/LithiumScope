"""Persistent training session logs, including stdout and failed preparation steps."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Iterator

from lithiumscope.core.logger import configure_logging
from lithiumscope.core.paths import LOGS_DIR


_ACTIVE_TRAINING_DIR: ContextVar[Path | None] = ContextVar(
    "active_training_dir", default=None,
)


def active_training_dir() -> Path | None:
    """Directory owning the active interactive training session, if any."""
    return _ACTIVE_TRAINING_DIR.get()


def register_training_run(model_group: str, run_dir: Path) -> None:
    """Link the training run and its state to the session-level logs."""
    active = active_training_dir()
    if active is None:
        return
    manifest = active / "runs.json"
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    entries[model_group] = str(run_dir)
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest)


def training_run_states(directory: Path) -> dict[str, dict]:
    manifest = directory / "runs.json"
    if not manifest.exists():
        return {}
    states = {}
    for model, run_dir in json.loads(manifest.read_text(encoding="utf-8")).items():
        run_file = Path(run_dir) / "run.json"
        try:
            data = json.loads(run_file.read_text(encoding="utf-8"))
            artifacts = {
                "ranking": Path(run_dir) / "tables/competition_ranking.csv",
                "baseline_folds": Path(run_dir) / "tables/baseline_fold_metrics.csv",
                "scientific_diagnostics": Path(run_dir) / "scientific_diagnostics/scientific_diagnostics.json",
                "scientific_group_metrics": Path(run_dir) / "scientific_diagnostics/per_spatial_group.csv",
                "scientific_fold_manifest": Path(run_dir) / "manifests/scientific_folds.json",
                "dashboard": Path(run_dir) / "dashboard.html",
                "workbook": Path(run_dir) / "exports" / f"{model}_results.xlsx",
            }
            states[model] = {
                "run_dir": str(run_dir),
                "state": data["state"],
                "summary": data.get("summary", {}),
                "artifacts": {
                    key: str(path) for key, path in artifacts.items() if path.is_file()
                },
            }
        except (OSError, KeyError, ValueError):
            states[model] = {"run_dir": str(run_dir), "state": "unknown"}
    return states


class _MirrorStream:
    def __init__(self, original, output):
        self.original = original
        self.output = output
        self.encoding = getattr(original, "encoding", "utf-8")

    def write(self, value: str) -> int:
        self.original.write(value)
        # Spinner redraws are not durable progress records; keep finalized lines.
        if "\r" in value and "\n" not in value:
            return len(value)
        self.output.write(value.replace("\r", "\n"))
        self.output.flush()
        return len(value)

    def flush(self) -> None:
        self.original.flush()
        self.output.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())


@contextmanager
def training_session(choice: str) -> Iterator[Path]:
    """Store progress, exceptions and Python logger records in one run folder."""
    configure_logging()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = LOGS_DIR / "training" / f"session_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    logger = logging.getLogger("lithiumscope")
    status_file = directory / "status.json"
    def status(state: str, **extra) -> None:
        tmp = status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "schema_version": 1,
            "state": state,
            "choice": choice,
            "started_at_utc": started,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "console_log": str(directory / "console.log"),
            "events_log": str(directory / "events.log"),
            **extra,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(status_file)

    handler = logging.FileHandler(directory / "events.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logger.addHandler(handler)
    status("running")
    context_token = _ACTIVE_TRAINING_DIR.set(directory)
    with (directory / "console.log").open("a", encoding="utf-8", buffering=1) as output:
        stdout = _MirrorStream(sys.stdout, output)
        stderr = _MirrorStream(sys.stderr, output)
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                print(f"Registro de entrenamiento: {directory}")
                logger.info("TRAINING_SESSION_BEGIN choice=%s path=%s", choice, directory)
                yield directory
                states = training_run_states(directory)
                observed = {item["state"] for item in states.values()}
                final_state = (
                    "failed" if observed & {"failed", "crashed", "cancelled"}
                    else "partial" if observed & {"partial", "unknown"}
                    else "completed"
                )
                status(final_state, model_runs=states)
                logger.info("TRAINING_SESSION_END state=%s model_runs=%s", final_state, states)
        except BaseException as exc:
            status("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                   error_type=type(exc).__name__, error=str(exc))
            details = "".join(traceback.format_exception(exc))
            (directory / "failure_traceback.txt").write_text(details, encoding="utf-8")
            logger.error("Training session interrupted: %s", details)
            output.write(details)
            output.flush()
            raise
        finally:
            handler.flush()
            logger.removeHandler(handler)
            handler.close()
            _ACTIVE_TRAINING_DIR.reset(context_token)
