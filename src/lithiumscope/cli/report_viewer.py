from __future__ import annotations

from pathlib import Path
import webbrowser


def open_report_in_default_browser(path: Path) -> bool:
    """Open a local report using the operating system's default web browser."""
    target = path.resolve()
    if not target.is_file():
        return False
    try:
        return bool(webbrowser.open_new_tab(target.as_uri()))
    except Exception:
        return False
