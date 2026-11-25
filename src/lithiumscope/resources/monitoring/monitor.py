"""Per-operation resource evidence; one writer per uniquely identified file."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time

from lithiumscope.resources.detection.host import detect_gpus
from lithiumscope.resources.monitoring.sampler import ProcessTreeSampler


@contextmanager
def monitor_operation(path: Path, *, interval_seconds: float = 2.0, metadata=None):
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("Monitoring interval must be positive.")
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    lock = threading.Lock()
    start = time.monotonic()
    sample_count=0
    sampler=ProcessTreeSampler()
    peaks={}
    errors = []
    with path.open("x", encoding="utf-8", buffering=1) as stream:
        def emit(event, **values):
            record = {"event": event, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": time.monotonic() - start, **values}
            with lock:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        def capture():
            nonlocal sample_count
            try:
                sample = {"pid":os.getpid(),**sampler.sample(),"gpus":list(detect_gpus())}
                sample_count+=1
                for key in ("process_rss_bytes","tree_rss_bytes","cpu_percent","process_cpu_percent","tree_cpu_percent","ram_percent"):
                    value=sample.get(key)
                    if value is not None:
                        peaks[key]=max(peaks.get(key,value),value)
                for g in sample["gpus"]:
                    value=g.get("utilization_percent")
                    if value is not None:
                        peaks["gpu_percent"]=max(peaks.get("gpu_percent",value),value)
                    total,free=g.get("vram_total_bytes"),g.get("vram_available_bytes")
                    if total is not None and free is not None and 0<=free<=total and total>0:
                        used=total-free
                        peaks["vram_used_bytes"]=max(peaks.get("vram_used_bytes",used),used)
                        percent=100*used/total
                        peaks["vram_percent"]=max(peaks.get("vram_percent",percent),percent)
                emit("sample", **sample)
            except Exception as exc:
                errors.append(str(exc))
                emit("monitor_warning", error=str(exc))

        def worker():
            while not stop.wait(interval_seconds):
                capture()

        emit("start", pid=os.getpid(), interval_seconds=interval_seconds, metadata=metadata or {},
             measurement_note="Tree RSS is summed RSS (shared pages may be counted twice); CPU excludes children that exit between samples.")
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
            emit("finish", state=state, duration_seconds=time.monotonic()-start,
                 samples=sample_count, peak_rss_bytes=peaks.get("process_rss_bytes"),
                 peak_tree_rss_bytes=peaks.get("tree_rss_bytes"),
                 peak_cpu_percent=peaks.get("cpu_percent"),
                 peak_process_cpu_percent=peaks.get("process_cpu_percent"),
                 peak_tree_cpu_percent=peaks.get("tree_cpu_percent"),
                 peak_ram_percent=peaks.get("ram_percent"),peak_gpu_percent=peaks.get("gpu_percent"),
                 peak_vram_used_bytes=peaks.get("vram_used_bytes"),peak_vram_percent=peaks.get("vram_percent"),
                 warnings=errors,
                 saturation_signals=[name for name in ("cpu","ram","gpu","vram") if peaks.get(name+"_percent",0)>=95])
