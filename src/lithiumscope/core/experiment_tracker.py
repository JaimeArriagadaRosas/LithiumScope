from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.core.states import RunState


class ExperimentTracker:
    def __init__(self, run_dir: Path, run_id: str, model_group: str) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "run.json"
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "model_group": model_group,
            "state": RunState.CREATED.value,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": None,
            "runtime": runtime_fingerprint(),
            "events": [],
            "summary": {},
        }
        self._write()

    @property
    def state(self) -> RunState:
        return RunState(self.payload["state"])

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def set_state(self, state: RunState, **summary: Any) -> None:
        self.payload["state"] = state.value
        if summary:
            self.payload["summary"].update(summary)
        if state in {
            RunState.COMPLETED,
            RunState.PARTIAL,
            RunState.FAILED,
            RunState.CANCELLED,
        }:
            self.payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._write()

    def cancel_if_active(self) -> None:
        if self.state in {RunState.CREATED, RunState.RUNNING}:
            self.set_state(RunState.CANCELLED)

    def event(self, event: str, **data: Any) -> None:
        self.payload["events"].append(
            {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **data,
            }
        )
        self._write()

    def attach_summary(self, **summary: Any) -> None:
        self.payload["summary"].update(summary)
        self._write()
