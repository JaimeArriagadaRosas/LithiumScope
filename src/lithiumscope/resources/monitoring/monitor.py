"""Per-operation resource evidence; one writer per uniquely identified file."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

from lithiumscope.resources.detection.host import detect_gpus


def _sample() -> dict:
    result = {"pid": os.getpid(), "cpu_percent": None, "process_cpu_percent": None,
              "ram_available_bytes": None, "ram_percent": None, "process_rss_bytes": None,
              "child_processes": None, "gpus": list(detect_gpus())}
    try:
        import psutil
        process = psutil.Process()
        ram = psutil.virtual_memory()
        children = process.children(recursive=True)
        result.update(cpu_percent=psutil.cpu_percent(interval=None),
                      process_cpu_percent=process.cpu_percent(interval=None),
                      ram_available_bytes=int(ram.available), ram_percent=ram.percent,
                      process_rss_bytes=int(process.memory_info().rss), child_processes=len(children))
    except (ImportError, OSError):
        pass
    return result


@contextmanager
def monitor_operation(path: Path, *, interval_seconds: float = 2.0):
    if interval_seconds <= 0:
        raise ValueError("Monitoring interval must be positive.")
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    lock = threading.Lock()
    start = time.monotonic()
    samples = []
    errors = []
    with path.open("x", encoding="utf-8", buffering=1) as stream:
        def emit(event, **values):
            record = {"event": event, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": time.monotonic() - start, **values}
            with lock:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        def capture():
            try:
                sample = _sample()
                samples.append(sample)
                emit("sample", **sample)
            except Exception as exc:
                errors.append(str(exc))
                emit("monitor_warning", error=str(exc))

        def worker():
            while not stop.wait(interval_seconds):
                capture()

        emit("start", pid=os.getpid(), interval_seconds=interval_seconds)
        capture()
        thread = threading.Thread(target=worker, daemon=True, name="lithiumscope-resource-monitor")
        thread.start()
        state = "completed"
        try:
            yield
        except BaseException:
            state = "interrupted"
            raise
        finally:
            stop.set()
            thread.join()
            capture()
            rss = [s["process_rss_bytes"] for s in samples if s["process_rss_bytes"] is not None]
            cpu = [s["cpu_percent"] for s in samples if s["cpu_percent"] is not None]
            ram = [s["ram_percent"] for s in samples if s["ram_percent"] is not None]
            gpu = [g["utilization_percent"] for s in samples for g in s["gpus"]
                   if g.get("utilization_percent") is not None]
            emit("finish", state=state, duration_seconds=time.monotonic() - start,
                 samples=len(samples), peak_rss_bytes=max(rss, default=None),
                 peak_cpu_percent=max(cpu, default=None), peak_ram_percent=max(ram, default=None),
                 peak_gpu_percent=max(gpu, default=None), warnings=errors,
                 saturation_signals=[name for name, values in (("cpu", cpu), ("ram", ram), ("gpu", gpu))
                                     if values and max(values) >= 95])
