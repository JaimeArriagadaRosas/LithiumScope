import time

import pytest

from lithiumscope.resources.allocation.runtime import bounded_workers, bounded_cache_mb
from lithiumscope.resources.monitoring.sampler import ProcessTreeSampler


def test_budget_caps_overrides():
    assert bounded_workers(-1,6)==6
    assert bounded_workers(100,6)==6
    assert bounded_workers(2,6)==2
    with pytest.raises(ValueError):
        bounded_workers(0,6)
    assert bounded_cache_mb(200,80*1024**2)==10
    assert bounded_cache_mb(400,None)==200
    with pytest.raises(ValueError):
        bounded_cache_mb(200,1024)


def test_sampler_returns_interval_cpu_and_tree_memory():
    pytest.importorskip("psutil")
    sampler=ProcessTreeSampler()
    first=sampler.sample()
    assert first["process_cpu_percent"] is None
    assert first["tree_rss_bytes"]>=first["process_rss_bytes"]>0
    stop=time.monotonic()+.05
    while time.monotonic()<stop:
        sum(range(1000))
    second=sampler.sample()
    assert second["process_cpu_percent"] is not None
    assert second["tree_cpu_percent"]>=second["process_cpu_percent"]>=0
    assert second["measured_processes"]>=1


def test_canonical_estimators_respect_worker_and_spatial_policy(monkeypatch):
    from lithiumscope.resources import manager
    from lithiumscope.model_1.training.algorithms import random_forest,hist_gradient_boosting,svm
    from lithiumscope.model_2.training.algorithms import extra_trees,hist_gradient_boosting as hgb2
    monkeypatch.setattr(manager,"cpu_worker_budget",lambda:2)
    assert random_forest.create_model(n_jobs=-1).n_jobs==2
    assert extra_trees.create_model(n_jobs=20).n_jobs==2
    assert not hist_gradient_boosting.create_model().early_stopping
    assert not hgb2.create_model().early_stopping
    with pytest.raises(ValueError,match="random internal"):
        hist_gradient_boosting.create_model(early_stopping=True)
    monkeypatch.setattr(manager,"svm_cache_limit",lambda _:16.)
    assert svm.create_model(cache_size=1000).cache_size==16.
