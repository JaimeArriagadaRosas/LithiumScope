from __future__ import annotations

import atexit
import ctypes
import os
import signal
import subprocess
import threading
import time
from typing import Callable

from lithiumscope.core.logger import finalize_logging, get_logger

logger = get_logger("runtime.shutdown")


class GracefulExit(SystemExit):
    pass


class GracefulShutdownManager:
    def __init__(self, child_timeout_seconds: float = 2.0) -> None:
        self.child_timeout_seconds = child_timeout_seconds
        self._lock = threading.RLock()
        self._requested = threading.Event()
        self._children: set[subprocess.Popen] = set()
        self._callbacks: list[Callable[[], None]] = []
        self._installed = False
        self._finalized = False
        self._windows_handler = None

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def register_process(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._children.add(process)

    def unregister_process(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._children.discard(process)

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def install(self) -> None:
        if self._installed:
            return
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._signal_handler)
        if os.name == "nt":
            self._install_windows_console_handler()
        atexit.register(self.finalize)
        self._installed = True
        logger.info("Graceful shutdown handlers installed")

    def _signal_handler(self, signum, _frame) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        self.request_shutdown(f"signal:{name}", exit_code=130)
        raise GracefulExit(130)

    def _install_windows_console_handler(self) -> None:
        handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

        @handler_type
        def handler(ctrl_type: int) -> bool:
            if ctrl_type in {2, 5, 6}:
                self.request_shutdown(f"windows_console_event:{ctrl_type}", exit_code=0)
                return True
            return False

        kernel32 = ctypes.windll.kernel32
        if kernel32.SetConsoleCtrlHandler(handler, True):
            self._windows_handler = handler
            logger.info("Windows console close handler installed")
        else:
            logger.warning("Windows SetConsoleCtrlHandler could not be installed")

    def _stop_children(self) -> None:
        with self._lock:
            children = [child for child in self._children if child.poll() is None]
        for child in children:
            try:
                logger.info("Terminating child process pid=%s", child.pid)
                child.terminate()
            except OSError:
                logger.debug("Child process already terminated pid=%s", child.pid)
        deadline = time.monotonic() + self.child_timeout_seconds
        for child in children:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    logger.warning("Killing unresponsive child process pid=%s", child.pid)
                    child.kill()
                except OSError:
                    pass

    def request_shutdown(self, reason: str, exit_code: int = 0) -> None:
        with self._lock:
            if self._requested.is_set():
                return
            self._requested.set()
        logger.info("Shutdown requested reason=%s exit_code=%d", reason, exit_code)
        if reason != "normal":
            print(f"\nCerrando LithiumScope de forma segura ({reason})...")
        self._stop_children()
        with self._lock:
            callbacks = list(reversed(self._callbacks))
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Shutdown cleanup callback failed")

    def finalize(self) -> None:
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
        if not self._requested.is_set():
            self.request_shutdown("normal", 0)
        self._stop_children()
        logger.info("LithiumScope runtime finalized")
        finalize_logging()


_MANAGER = GracefulShutdownManager()


def get_shutdown_manager() -> GracefulShutdownManager:
    return _MANAGER
