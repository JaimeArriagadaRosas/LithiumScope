from __future__ import annotations

import sys
import threading
import time
from typing import TextIO


class Spinner:
    """Single-line console spinner that rewrites its current status in-place."""

    _FRAMES = ("|", "/", "-", "\\")

    def __init__(
        self,
        message: str,
        *,
        interval: float = 0.12,
        stream: TextIO | None = None,
    ) -> None:
        self._message = message
        self._interval = interval
        self._stream = stream or sys.stdout
        self._enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._last_width = 0
        self._started = False

    def start(self) -> "Spinner":
        if self._started:
            return self
        self._started = True
        if not self._enabled:
            self._stream.write(self._message + "\n")
            self._stream.flush()
            return self

        self._thread = threading.Thread(
            target=self._animate,
            name="lithiumscope-spinner",
            daemon=True,
        )
        self._thread.start()
        return self

    def update(self, message: str) -> None:
        with self._lock:
            self._message = message

    def _animate(self) -> None:
        index = 0
        while not self._stop.wait(self._interval):
            with self._lock:
                message = self._message
            frame = self._FRAMES[index % len(self._FRAMES)]
            index += 1
            self._rewrite(f"{frame} {message}")

    def _rewrite(self, text: str) -> None:
        width = max(self._last_width, len(text))
        self._stream.write("\r" + text.ljust(width))
        self._stream.flush()
        self._last_width = width

    def _finish(self, marker: str, message: str | None) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.5, self._interval * 4))

        final_message = message or self._message
        if self._enabled:
            self._rewrite(f"{marker} {final_message}")
            self._stream.write("\n")
        else:
            self._stream.write(f"{marker} {final_message}\n")
        self._stream.flush()

    def succeed(self, message: str | None = None) -> None:
        self._finish("[OK]", message)

    def fail(self, message: str | None = None) -> None:
        self._finish("[ERROR]", message)

    def stop(self, message: str | None = None) -> None:
        self._finish("[--]", message)
