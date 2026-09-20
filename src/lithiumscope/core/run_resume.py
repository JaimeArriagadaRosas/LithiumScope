from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash

_RESUMABLE_STATES = {"cancelled", "partial", "running", "crashed", "completed"}


def _atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == 259
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
        payload.setdefault("summary", {})["crash_reason"] = "Previous process ended without graceful finalization."
        payload.setdefault("events", []).append({
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "event": "run_recovered_as_crashed",
            "previous_state": "running",
        })
        _atomic_write(run_file, payload)
        recovered.append(run_file.parent)
    return recovered


def _scientific_config_payload(model_group: str, config_names: tuple[str, ...]) -> dict:
    payload: dict = {"model_group": model_group}
    for name in config_names:
        config = load_config(name)
        if name == "app":
            app = config.get("app", {})
            payload[name] = {
                "prefer_gpu": app.get("prefer_gpu"),
                "random_seed": app.get("random_seed"),
                "reproducibility": config.get("reproducibility", {}),
                "resources": config.get("resources", {}),
            }
        else:
            payload[name] = config
    return payload


def _scientific_code_files(model_group: str) -> list[Path]:
    source_root = PROJECT_ROOT / "src" / "lithiumscope"
    model_root = source_root / model_group
    files: list[Path] = []
    for filename in ("pipeline.py", "schema.py"):
        candidate = model_root / filename
        if candidate.exists():
            files.append(candidate)
    for directory_name in ("steps", "training"):
        directory = model_root / directory_name
        if directory.exists():
            files.extend(directory.rglob("*.py"))
    if model_group == "model_2":
        directory = model_root / "data"
        if directory.exists():
            files.extend(directory.rglob("*.py"))
    for filename in ("evaluation/metrics.py", "evaluation/baseline.py"):
        candidate = model_root / filename
        if candidate.exists():
            files.append(candidate)
    for path in (
        source_root / "core" / "device.py",
        source_root / "core" / "resources.py",
        source_root / "core" / "reproducibility.py",
        source_root / "core" / "scientific_checks.py",
    ):
        if path.exists():
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def _scientific_code_fingerprint(model_group: str) -> str:
    payload = {
        path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path)
        for path in _scientific_code_files(model_group)
    }
    return canonical_json_hash(payload)


def build_training_signature(model_group: str, dataset_path: Path, config_names: tuple[str, ...]) -> str:
    payload = {
        "schema_version": 2,
        "model_group": model_group,
        "dataset_sha256": file_sha256(dataset_path),
        "scientific_config": _scientific_config_payload(model_group, config_names),
        "scientific_code_sha256": _scientific_code_fingerprint(model_group),
    }
    return canonical_json_hash(payload)


def find_compatible_run(model_group: str, signature: str) -> Path | None:
    recover_abandoned_runs()
    directory = RESULTS_DIR / model_group / "runs"
    if not directory.exists():
        return None
    for run_dir in sorted((p for p in directory.iterdir() if p.is_dir()), reverse=True):
        run_file = run_dir / "run.json"
        if not run_file.exists():
            continue
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("summary", {}).get("training_signature") == signature and str(payload.get("state", "")) in _RESUMABLE_STATES:
            return run_dir
    return None
