"""Runtime worker/cache controls for canonical estimators."""
from __future__ import annotations

from contextlib import contextmanager
import math
import sys


def bounded_workers(requested, budget):
    if type(budget) is not int or budget < 1:
        raise ValueError("Invalid worker budget")
    if requested is None or requested == -1:
        return budget
    if type(requested) is not int or requested < 1:
        raise ValueError("Worker count must be a positive integer or -1")
    return min(requested,budget)


def bounded_cache_mb(requested, ram_budget_bytes):
    if not isinstance(requested,(int,float)) or isinstance(requested,bool) or not math.isfinite(requested) or requested<=0:
        raise ValueError("Invalid SVM cache request")
    # The cache is only one component of total estimator memory.
    if ram_budget_bytes is None:
        return min(200.,float(requested))
    limit=ram_budget_bytes/(8*1024**2)
    if limit<1:
        raise ValueError("Insufficient RAM budget for SVM kernel cache")
    return min(float(requested),limit)


@contextmanager
def native_thread_limits(threads):
    """Constrain supported native libraries and restore process-global settings."""
    from threadpoolctl import threadpool_limits
    torch=sys.modules.get("torch")
    previous=None
    with threadpool_limits(limits=threads):
        try:
            if torch is not None and callable(getattr(torch,"get_num_threads",None)):
                previous=torch.get_num_threads()
                torch.set_num_threads(threads)
            yield
        finally:
            if previous is not None:
                torch.set_num_threads(previous)
