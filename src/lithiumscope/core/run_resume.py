from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import CONFIG_DIR, RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash, runtime_fingerprint


_RESUMABLE_STATES = {
    "cancelled",
    "partial",
    "running",
    "crashed",
    "completed",
}


def _atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            ):
                return False
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def recover_abandoned_runs(results_root: Path | None = None) -> list[Path]:
    """Mark orphaned RUNNING runs as CRASHED without discarding checkpoints."""
    root = results_root or RESULTS_DIR
    recovered: list[Path] = []
    if not root.exists():
        return recovered

    current_host = platform.node()
    for run_file in root.glob("model_*/runs/*/run.json"):
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if payload.get("state") != "running":
            continue

        runtime = payload.get("runtime", {})
        active_process = payload.get("active_process") or {}
        owner_pid = active_process.get("pid", runtime.get("pid"))
        owner_host = active_process.get("hostname", runtime.get("hostname"))

        # Legacy RUNNING records have no process identity. After a fresh
        # application start they are considered abandoned.
        abandoned = owner_pid is None
        if owner_pid is not None and (not owner_host or owner_host == current_host):
            try:
                abandoned = not _pid_alive(int(owner_pid))
            except (TypeError, ValueError):
                abandoned = True

        if not abandoned:
            continue

        payload["state"] = "crashed"
        payload["active_process"] = None
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload.setdefault("summary", {})["crash_reason"] = (
            "Previous process ended without graceful finalization."
        )
        payload.setdefault("events", []).append(
            {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "event": "run_recovered_as_crashed",
                "previous_state": "running",
            }
        )
        _atomic_write(run_file, payload)
        recovered.append(run_file.parent)

    return recovered


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
    recover_abandoned_runs()
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
        if state in _RESUMABLE_STATES:
            return run_dir
    return None
