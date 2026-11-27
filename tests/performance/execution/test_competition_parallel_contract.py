from types import SimpleNamespace
import pytest
from lithiumscope.resources.allocation.competition_workers import enabled,candidate_result,schedule_candidates
from lithiumscope.resources.allocation.policy import ResourcePolicy

def device(**kw):
    return SimpleNamespace(**({"is_gpu":False,"any_cuda_backend":False,"mps_available":False}|kw))

def test_cpu_parallelism_explicit_and_gpu_keeps_serial(tmp_path):
    assert not enabled(device(),ResourcePolicy())
    assert enabled(device(),ResourcePolicy(max_parallel_tasks=2))
    assert not enabled(device(any_cuda_backend=True),ResourcePolicy(max_parallel_tasks=2))
    assert schedule_candidates("model_1",[],tmp_path,device(),policy=ResourcePolicy()) is None
    assert not list(tmp_path.iterdir())

def test_worker_error_not_replaced_by_unverified_retry():
    with pytest.raises(RuntimeError,match="failed"):
        candidate_result({"rf":RuntimeError("failed")},"rf")
