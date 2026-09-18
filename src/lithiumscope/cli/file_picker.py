from __future__ import annotations

from pathlib import Path
from typing import Iterable

from lithiumscope.core.logger import get_logger

logger = get_logger("file_picker")


def pick_file(
    title: str,
    patterns: Iterable[tuple[str, str]],
) -> Path | None:
    """Open the operating-system file picker; fall back to a text path if unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(title=title, filetypes=list(patterns))
        root.destroy()
        if selected:
            path = Path(selected)
            logger.info("File selected: %s", path)
            return path
        return None
    except Exception:
        logger.warning("Graphical file picker unavailable; using console fallback", exc_info=True)

    try:
        raw = input("Ruta del archivo (vacío para cancelar): ").strip()
    except EOFError:
        return None
    return Path(raw) if raw else None
