"""Verify that the descriptive cache uses exactly the selected development protocol."""
import json

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_1.eda import pipeline as m1
from lithiumscope.model_2.eda import pipeline as m2
from lithiumscope.model_2.validation.spatial import make_scientific_folds


def test_m1_uses_prepared_identity_and_reuses_exact_folds(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    frame = pd.DataFrame({
        "Li_icpms": np.arange(12, dtype=float)+1,
        "SUM (no water)": [98.]*12, "SiO2": np.arange(12, dtype=float)+50,
        "Longitude": np.repeat([-70., -71., -72., -73., -74., -75.], 2),
        "Latitude": [-20.]*12,
    })
    frame.to_csv(path, index=False)
    prepared = m1.prepare_training_data(path)
    cfg = m1.load_config("model_1")
    prepared.outer_splits, prepared.validation_strategy = m1.materialize_regression_splits(
        prepared.x, prepared.y, prepared.groups, 5, 42, require_groups=True)
    monkeypatch.setattr(m1, "prepare_training_data", lambda *_: pytest.fail("prepared twice"))
    kwargs = {"prepared": prepared, "config": cfg, "output_root": tmp_path / "eda"}
    first = m1.run_eda(path, **kwargs)
    assert m1.run_eda(path, **kwargs) == first
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["development_protocol"]["fold_count"] == 5
    prepared.outer_splits = list(reversed(prepared.outer_splits))
    second = m1.run_eda(path, **kwargs)
    assert first != second


def test_m2_uses_exact_folds_and_quality_without_reloading(tmp_path, monkeypatch):
    path = tmp_path / "manifest.csv"
    frame = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(12)],
        "Li_icpms": [1., 3., 8., 12.]*3, "spatial_group": np.repeat(["a", "b", "c"], 4),
        "longitude": np.repeat([-70., -71., -72.], 4), "latitude": [-20.]*12,
    })
    frame.to_csv(path, index=False)
    features = frame[["Li_icpms", "spatial_group"]].assign(red_mean=np.arange(12, dtype=float))
    cfg = {"training": {"lithium_column": "Li_icpms", "spatial_group_column": "spatial_group"},
           "model": {"high_lithium_quantile": .75}, "imagery": {"bands": ["red"]},
           "validation": {"folds": 3}}
    keep = np.ones(12, bool)
    keep[0] = False
    monkeypatch.setattr(m2, "patch_quality", lambda *_: (pd.DataFrame({"train_eligible": keep}), {}))
    monkeypatch.setattr(m2, "load_model_2_training_frame", lambda *_: pytest.fail("frame reloaded"))
    folds = make_scientific_folds(frame.Li_icpms, frame.spatial_group, n_splits=3)
    kwargs = {"frame": features, "config": cfg, "scientific_folds": folds,
              "quality_mask": keep, "output_root": tmp_path / "eda"}
    first = m2.run_eda(path, **kwargs)
    assert m2.run_eda(path, **kwargs) == first
    summary = pd.read_csv(first / "tables/outer_train_summary.csv")
    assert summary.fit_eligible_rows.sum() == 22
    with pytest.raises(ValueError, match="QA disagree"):
        m2.run_eda(path, **{**kwargs, "quality_mask": np.ones(12, bool)})
