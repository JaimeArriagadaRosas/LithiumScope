"""Guardrails for M2 v4; no remote imagery downloads in unit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.methodology.rededge_v4 import (
    TOTAL_FEATURES,
    NEW_FEATURES,
    NEW_BANDS,
    reflectance_from_dn,
    extract_rededge_v4,
    append_rededge_features,
)
from lithiumscope.model_2.methodology import rededge_acquisition as ac
from scripts.scientific import m2_rededge_v4_build as builder
from scripts.scientific import m2_rededge_v4_competition as competition
from scripts.scientific.m2_rededge_v4_report import paired_group_ci


def test_exact_44_new_features_and_base_126_unchanged():
    cube = np.stack([np.ones((32, 32), np.float32) * (0.15 + i * 0.03) for i in range(4)])
    extra, coverage = extract_rededge_v4(cube)
    assert len(extra) == NEW_FEATURES == 44
    assert coverage == 1
    assert set(extra) == set(extract_rededge_v4(cube)[0])
    assert "ndre1_refl_mean" in extra and "ndre2_refl_p90" in extra
    initial = {f"old_{i}": float(i) for i in range(126)}
    combo = append_rededge_features(initial, extra)
    assert len(combo) == TOTAL_FEATURES == 170
    assert list(combo)[:126] == list(initial)
    assert all(combo[k] == v for k, v in initial.items())


def test_radiometry_by_band_and_nodata_preservation():
    dn = np.full((4, 3, 3), 1100.0, np.float32)
    dn[1, 0, 0] = np.nan
    scales = ((0.0001, -0.1), (0.0001, 0), (0.0002, -0.1), (0.0001, -0.1))
    out = reflectance_from_dn(dn, scales)
    assert out.shape == dn.shape
    assert np.isnan(out[1, 0, 0])
    assert out[0, 0, 0] == pytest.approx(0.01)
    assert out[1, 1, 1] == pytest.approx(0.11)
    assert out[2, 1, 1] == pytest.approx(0.12)
    with pytest.raises(ValueError):
        reflectance_from_dn(dn, ((0.0001, -0.1),) * 3)


def test_invalid_partial_coverage_not_mislabeled():
    empty = np.full((4, 32, 32), np.nan, np.float32)
    with pytest.raises(ValueError, match="coverage"):
        extract_rededge_v4(empty)
    with pytest.raises(ValueError, match="cutoff"):
        extract_rededge_v4(np.ones((4, 32, 32)), min_joint_valid=1.1)


def test_scene_asset_names_and_scaling_from_saved_stac(tmp_path):
    scene = "S2A_TEST_20260101_0_L2A"
    assets = {}
    for i, name in enumerate(NEW_BANDS):
        assets[name] = {
            "href": f"https://example.test/{name}.tif",
            "raster:bands": [{"scale": 0.0001, "offset": -0.1 + i * 0.01, "nodata": 0}],
        }
    (tmp_path / f"{scene}.json").write_text(
        json.dumps({"id": scene, "collection": "sentinel-2-l2a", "assets": assets}),
        encoding="utf8",
    )
    descriptor, digest = ac.scene_item(scene, cache=tmp_path)
    assert len(digest) == 64
    assert tuple(descriptor["bands"]) == NEW_BANDS
    assert descriptor["bands"]["nir08"]["offset"] == pytest.approx(-0.07)


def test_patch_cache_fingerprint_refuses_original_drift(monkeypatch, tmp_path):
    original = tmp_path / "original.npy"
    np.save(original, np.ones((6, 32, 32), np.float32))
    record = SimpleNamespace(
        sample_id="sample-01",
        image_path=str(original),
        sentinel_scene_id="S2A_TEST",
        latitude=-16.5,
        longitude=-70.1,
    )
    spec = {
        name: {"href": f"https://example.test/{name}.tif", "scale": 0.0001, "offset": -0.1, "nodata": 0}
        for name in NEW_BANDS
    }
    monkeypatch.setattr(ac, "scene_item", lambda scene: ({"bands": spec}, "somehash"))
    reads = []

    def mock_reader(href, lat, lon, size, pixels):
        reads.append(href)
        return np.full((pixels, pixels), 1500.0, np.float32)

    monkeypatch.setattr(ac, "_read_patch", mock_reader)
    out = tmp_path / "cache"
    d, a = ac.cached_rededge_sample(record, output=out)
    assert len(reads) == 4 and len(d) == 44 and a["rededge_joint_valid_fraction"] == 1
    ac.cached_rededge_sample(record, output=out)
    assert len(reads) == 4
    np.save(original, np.full((6, 32, 32), 2.0, np.float32))
    with pytest.raises(ValueError, match="drift"):
        ac.cached_rededge_sample(record, output=out)


def test_partial_real_matrix_cannot_be_competed(monkeypatch, tmp_path):
    root = tmp_path
    src = root / "data/processed/model_2"
    src.mkdir(parents=True)
    n = 706
    manifest = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(n)],
            "Li_icpms": np.arange(n, dtype=float),
            "spatial_group": [f"g{i // 8}" for i in range(n)],
            "latitude": -16.0,
            "longitude": -70.0,
            "sentinel_scene_id": "S2A_SAMPLE",
            "image_path": "example.npy",
        }
    )
    manifest.to_csv(src / "training_manifest.csv", index=False)
    base = pd.DataFrame(
        {
            "Li_icpms": manifest.Li_icpms,
            "spatial_group": manifest.spatial_group,
            **{f"old{i}": np.full(n, i, dtype=float) for i in range(126)},
        }
    )
    monkeypatch.setattr(builder, "load_model_2_training_frame", lambda _: base)
    monkeypatch.setattr(
        builder,
        "cached_rededge_sample",
        lambda row, **_: (
            {f"redge_{i}": float(i) for i in range(44)},
            {"sample_id": str(row.sample_id), "rededge_joint_valid_fraction": 1.0},
        ),
    )
    state = builder.build(706, limit=3, root=root)
    assert state["status"] == "incomplete_no_training"
    assert state["downloaded"] == 3
    with pytest.raises(FileNotFoundError):
        competition.compare(706, root=root)
    done = builder.build(706, root=root)
    assert done["status"] == "complete"
    matrix = pd.read_csv(root / "results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/training_v4_170.csv")
    assert matrix.shape == (706, 172)
    assert np.array_equal(
        matrix[[f"old{i}" for i in range(126)]].to_numpy(float),
        base[[f"old{i}" for i in range(126)]].to_numpy(float),
    )


def test_grouped_paired_bootstrap_has_three_intervals():
    y = np.tile([0, 1, 0, 1], 10)
    old = np.tile([0.2, 0.8, 0.4, 0.6], 10)
    new = np.tile([0.1, 0.9, 0.35, 0.65], 10)
    groups = np.repeat(np.arange(10), 4).astype(str)
    result = paired_group_ci(y, old, new, groups, n=60)
    assert set(result) == {"delta_auc_ci95", "delta_ap_ci95", "delta_brier_ci95"}
    assert all(len(val) == 2 for val in result.values())


def test_v4_quality_uses_old_and_new_spectral_coverage(tmp_path):
    from lithiumscope.model_2.methodology.rededge_training import (
        combine_quality_masks,
    )

    m = pd.DataFrame({"sample_id": ["S1", "S2", "S3"]})
    path = tmp_path / "quality.csv"
    pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "rededge_joint_valid_fraction": [1.0, 0.6, 0.99],
        }
    ).to_csv(path, index=False)
    selected, audit = combine_quality_masks(
        np.array([True, True, False]),
        path,
        m,
    )
    assert selected.tolist() == [True, False, False]
    assert audit["joint_qa95_train_eligible"] == 1
    assert audit["all_outer_test_retained"] is True
    changed = pd.read_csv(path)
    changed.loc[1, "sample_id"] = "WRONG"
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="aligned"):
        combine_quality_masks(np.ones(3, bool), path, m)


@pytest.mark.parametrize(
    "family",
    (
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
        "catboost",
        "svm_rbf",
    ),
)
def test_all_six_families_can_fit_170_features(family):
    from lithiumscope.core.device import detect_device
    from lithiumscope.model_2.methodology.rededge_training import fit_frozen_model

    n = 90
    x = pd.DataFrame(
        np.random.default_rng(42).normal(size=(n, 170)),
        columns=[f"f{i}" for i in range(170)],
    )
    y = np.tile([0, 1, 0], n // 3)
    groups = pd.Series(np.repeat(np.arange(n // 3), 3))
    parameters = {
        "random_forest": {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 3},
        "extra_trees": {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 3},
        "hist_gradient_boosting": {"max_iter": 20, "min_samples_leaf": 5},
        "xgboost": {"n_estimators": 20, "max_depth": 2},
        "catboost": {"iterations": 20, "depth": 3},
        "svm_rbf": {"C": 1.0, "gamma": 0.01},
    }
    model = fit_frozen_model(
        family,
        parameters[family],
        x,
        y,
        groups,
        detect_device(prefer_gpu=False),
        42,
    )
    score = model.predict_proba(x.iloc[:6])[:, 1]
    assert len(score) == 6 and np.isfinite(score).all()
    assert np.all((score >= 0) & (score <= 1))


def test_nested_and_result_document_refuse_partial_data(tmp_path):
    from scripts.scientific.m2_rededge_v4_nested import train_all
    from scripts.scientific.m2_v4_document_result import document

    with pytest.raises(FileNotFoundError):
        train_all(706, root=tmp_path)
    with pytest.raises(FileNotFoundError):
        document("nested", root=tmp_path)


def test_new_site_data_effect_requires_verified_images(tmp_path):
    from scripts.scientific.m2_v4_new_site_effect import run

    with pytest.raises(FileNotFoundError):
        run(root=tmp_path)
