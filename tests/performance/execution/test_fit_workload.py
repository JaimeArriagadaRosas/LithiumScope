import json

import pandas as pd
import pytest

from lithiumscope.resources.allocation.estimates import estimate_fit_demand
from lithiumscope.resources.allocation.policy import ResourcePolicy
from lithiumscope.resources.detection.host import HostResources
from lithiumscope.resources import manager
from lithiumscope.resources.monitoring import monitor


def test_demand_accounts_for_encoded_features_and_attempts():
    x=pd.DataFrame({"x":[1.,2.,3.],"rock":["a","b","c"]})
    d=estimate_fit_demand(x,"svm",folds=5,inner_folds=3,trials=20,optimize=True,categorical=["rock"])
    assert d.encoded_features_upper_bound==6
    assert d.planned_inner_fits_upper_bound==360
    assert d.workload().tasks==1 and d.working_memory_estimate_bytes>d.matrix_bytes
    with pytest.raises(ValueError,match="schema"):
        estimate_fit_demand(x,"svm",folds=5,inner_folds=3,trials=20,optimize=True,categorical=["missing"])


def test_nested_scope_cannot_expand_parent_budget_and_restores_it(tmp_path,monkeypatch):
    host=HostResources(16,8,16,8*1024**3,4*1024**3,None,())
    monkeypatch.setattr(manager,"detect_host",lambda **kw:host)
    monkeypatch.setattr(manager,"load_policy",lambda:ResourcePolicy())
    demand=estimate_fit_demand(pd.DataFrame({"x":[1.,2.]}),"random_forest",
                              folds=3,inner_folds=2,trials=1,optimize=False)
    token=manager._active_threads.set(2)
    try:
        with manager.fit_workload_scope(demand,evidence_path=tmp_path/"plan.json") as plan:
            assert plan.threads_per_task==2
            assert manager.cpu_worker_budget()==2
        assert manager.cpu_worker_budget()==2
        assert json.loads((tmp_path/"plan.json").read_text())["estimate_is_hard_cap"] is False
    finally:
        manager._active_threads.reset(token)


def test_gpu_vram_peak_is_measured_and_not_invented(tmp_path,monkeypatch):
    monkeypatch.setattr(monitor,"detect_gpus",lambda:({"vram_total_bytes":1000,"vram_available_bytes":30,
                                                     "utilization_percent":40},))
    path=tmp_path/"monitor.jsonl"
    with monitor.monitor_operation(path,interval_seconds=.1):
        pass
    last=json.loads(path.read_text().splitlines()[-1])
    assert last["peak_vram_used_bytes"]==970 and last["peak_vram_percent"]==97.
    assert "vram" in last["saturation_signals"]
