from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
from typing import Iterable

from lithiumscope.core.logger import get_logger
from lithiumscope.runtime.graceful_shutdown import get_shutdown_manager

logger = get_logger("file_picker")


def _run_process(args: list[str]) -> str | None:
    manager = get_shutdown_manager()
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    manager.register_process(process)
    try:
        stdout, stderr = process.communicate()
    finally:
        manager.unregister_process(process)

    if process.returncode not in {0, 1}:
        logger.warning("Native file picker failed rc=%s stderr=%s", process.returncode, stderr.strip())
        return None
    selected = stdout.strip()
    return selected or None


def _windows_filter(patterns: Iterable[tuple[str, str]]) -> str:
    parts: list[str] = []
    for label, raw_pattern in patterns:
        normalized = ";".join(raw_pattern.split())
        parts.extend([f"{label} ({normalized})", normalized])
    return "|".join(parts)


def _pick_windows(title: str, patterns: Iterable[tuple[str, str]]) -> str | None:
    executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not executable:
        return None
    safe_title = title.replace("'", "''")
    safe_filter = _windows_filter(patterns).replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
        f"$dialog.Title = '{safe_title}'; "
        f"$dialog.Filter = '{safe_filter}'; "
        "$dialog.Multiselect = $false; "
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
        "{ [Console]::Write($dialog.FileName) }"
    )
    return _run_process([executable, "-NoProfile", "-STA", "-Command", script])


def _pick_macos(title: str) -> str | None:
    executable = shutil.which("osascript")
    if not executable:
        return None
    safe_title = title.replace('"', '\\"')
    script = f'POSIX path of (choose file with prompt "{safe_title}")'
    return _run_process([executable, "-e", script])


def _pick_linux(title: str, patterns: Iterable[tuple[str, str]]) -> str | None:
    zenity = shutil.which("zenity")
    if zenity:
        args = [zenity, "--file-selection", f"--title={title}"]
        for label, raw_pattern in patterns:
            args.append(f"--file-filter={label} | {raw_pattern}")
        return _run_process(args)

    kdialog = shutil.which("kdialog")
    if kdialog:
        joined = " ".join(pattern for _, pattern in patterns)
        return _run_process([kdialog, "--getopenfilename", ".", joined, "--title", title])
    return None


def _native_pick(title: str, patterns: Iterable[tuple[str, str]]) -> str | None:
    if os.name == "nt":
        return _pick_windows(title, patterns)
    if sys_platform() == "darwin":
        return _pick_macos(title)
    return _pick_linux(title, patterns)


def sys_platform() -> str:
    import sys

    return sys.platform


def pick_file(title: str, patterns: Iterable[tuple[str, str]]) -> Path | None:
    """Use an OS-native dialog without creating a Tkinter event loop."""
    patterns = list(patterns)
    try:
        selected = _native_pick(title, patterns)
        if selected:
            path = Path(selected)
            logger.info("File selected: %s", path)
            return path
        if selected == "":
            return None
    except Exception:
        logger.warning("Native file picker unavailable; using console fallback", exc_info=True)

    try:
        raw = input("Ruta del archivo (vacío para cancelar): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return Path(raw) if raw else None
