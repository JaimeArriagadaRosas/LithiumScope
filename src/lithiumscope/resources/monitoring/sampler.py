"""Stateful host/process-tree measurement; interval CPU, not a cold zero."""
from __future__ import annotations

import time


class ProcessTreeSampler:
    def __init__(self):
        self._previous = {}
        self._timestamp = None
        self._psutil = None
        self._process = None
        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process()
            psutil.cpu_percent(interval=None)
        except (ImportError, OSError):
            pass

    def sample(self):
        result = {"cpu_percent": None, "process_cpu_percent": None,
                  "tree_cpu_percent": None, "ram_available_bytes": None,
                  "ram_percent": None, "process_rss_bytes": None,
                  "tree_rss_bytes": None, "child_processes": None,
                  "measured_processes": None, "unavailable_processes": None}
        ps = self._psutil
        if ps is None or self._process is None:
            return result
        now = time.monotonic()
        try:
            ram = ps.virtual_memory()
            result.update(ram_available_bytes=int(ram.available), ram_percent=float(ram.percent))
            cpu = ps.cpu_percent(interval=None)
            if self._timestamp is not None:
                result["cpu_percent"] = float(cpu)
            processes = [self._process, *self._process.children(recursive=True)]
        except (ps.Error, OSError):
            return result
        current, rss = {}, {}
        unavailable = 0
        for process in processes:
            try:
                with process.oneshot():
                    identity = (process.pid, process.create_time())
                    times = process.cpu_times()
                    current[identity] = float(times.user + times.system)
                    rss[identity] = int(process.memory_info().rss)
            except (ps.Error, OSError):
                unavailable += 1
        parent = next((key for key in current if key[0] == self._process.pid), None)
        if parent is not None:
            result["process_rss_bytes"] = rss[parent]
        result.update(child_processes=len(processes)-1, measured_processes=len(current),
                      unavailable_processes=unavailable,
                      tree_rss_bytes=sum(rss.values()) if rss else None)
        if self._timestamp is not None and now > self._timestamp:
            elapsed = now-self._timestamp
            overlap = current.keys() & self._previous.keys()
            increments = {key: max(0.,current[key]-self._previous[key]) for key in overlap}
            if increments:
                result["tree_cpu_percent"] = 100*sum(increments.values())/elapsed
            if parent in increments:
                result["process_cpu_percent"] = 100*increments[parent]/elapsed
        self._previous, self._timestamp = current, now
        return result
