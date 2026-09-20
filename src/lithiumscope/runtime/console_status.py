from __future__ import annotations

import shutil
import sys
import threading
from typing import TextIO


class ConsoleStatusManager:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout
        self._lock = threading.RLock()
        self._active: Spinner | None = None

    def _truncate(self, text: str) -> str:
        width = max(40, shutil.get_terminal_size(fallback=(120, 24)).columns) - 1
        return text if len(text) <= width else text[: max(1, width - 1)] + "…"

    def register(self, spinner: "Spinner") -> None:
        with self._lock:
            self._active = spinner

    def unregister(self, spinner: "Spinner") -> None:
        with self._lock:
            if self._active is spinner:
                self._active = None

    def rewrite(self, spinner: "Spinner", text: str) -> None:
        with self._lock:
            if self._active is spinner:
                spinner._rewrite_unlocked(self._truncate(text))

    def message(self, text: str) -> None:
        with self._lock:
            active = self._active
            if active is not None:
                active._clear_unlocked()
            self._stream.write(text.rstrip() + "\n")
            self._stream.flush()
            if active is not None:
                active._rewrite_unlocked(self._truncate(active._display_text()))


_DEFAULT_MANAGER = ConsoleStatusManager()


def console_message(text: str) -> None:
    _DEFAULT_MANAGER.message(text)


class Spinner:
    _FRAMES = ("|", "/", "-", "\\")

    def __init__(self, message: str, *, interval: float = 0.25, stream: TextIO | None = None) -> None:
        self._message = message
        self._interval = max(0.20, float(interval))
        self._stream = stream or sys.stdout
        self._enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._last_width = 0
        self._started = False
        self._frame = self._FRAMES[0]
        self._manager = _DEFAULT_MANAGER if self._stream is sys.stdout else ConsoleStatusManager(self._stream)

    def start(self) -> "Spinner":
        if self._started:
            return self
        self._started = True
        if not self._enabled:
            self._stream.write(self._message + "\n")
            self._stream.flush()
            return self
        self._manager.register(self)
        self._thread = threading.Thread(target=self._animate, name="lithiumscope-spinner", daemon=True)
        self._thread.start()
        return self

    def update(self, message: str) -> None:
        with self._lock:
            self._message = message

    def _display_text(self) -> str:
        with self._lock:
            return f"{self._frame} {self._message}"

    def _animate(self) -> None:
        index = 0
        while not self._stop.wait(self._interval):
            with self._lock:
                self._frame = self._FRAMES[index % len(self._FRAMES)]
                index += 1
                text = f"{self._frame} {self._message}"
            self._manager.rewrite(self, text)

    def _rewrite_unlocked(self, text: str) -> None:
        width = max(self._last_width, len(text))
        self._stream.write("\r" + text.ljust(width))
        self._stream.flush()
        self._last_width = width

    def _clear_unlocked(self) -> None:
        if self._last_width > 0:
            self._stream.write("\r" + (" " * self._last_width) + "\r")
            self._stream.flush()

    def _finish(self, marker: str, message: str | None) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.5, self._interval * 4))
        final_message = message or self._message
        if self._enabled:
            self._manager.rewrite(self, f"{marker} {final_message}")
            self._stream.write("\n")
            self._stream.flush()
            self._manager.unregister(self)
        else:
            self._stream.write(f"{marker} {final_message}\n")
            self._stream.flush()

    def succeed(self, message: str | None = None) -> None:
        self._finish("[OK]", message)

    def fail(self, message: str | None = None) -> None:
        self._finish("[ERROR]", message)

    def stop(self, message: str | None = None) -> None:
        self._finish("[--]", message)
