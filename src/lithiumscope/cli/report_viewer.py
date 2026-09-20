from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import webbrowser


def open_report_in_default_browser(path: Path) -> bool:
    target = path.resolve()
    if not target.is_file():
        return False

    try:
        if webbrowser.open_new_tab(target.as_uri()):
            return True
    except Exception:
        pass

    system = platform.system().lower()
    try:
        if system == "windows":
            os.startfile(str(target))  # type: ignore[attr-defined]
            return True
        if system == "darwin":
            subprocess.Popen(["open", str(target)])
            return True
        subprocess.Popen(["xdg-open", str(target)])
        return True
    except Exception:
        return False
