from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_2.texture_study import STATS, TEXTURE, texture_arms
from lithiumscope.model_2.evaluation.geographic_baseline import evaluate_geographic_baseline
from lithiumscope.model_2.preparation.features.rededge import NEW_BANDS
from lithiumscope.model_2.preparation.features.spectral import DEFAULT_BAND_NAMES


def test_texture_arms_change_only_texture_columns():
    prefixes = (*DEFAULT_BAND_NAMES, *(f"{band}_refl" for band in NEW_BANDS))
    cols = [f"{band}_{stat}" for band in prefixes for stat in (*STATS, *TEXTURE)]
    x = pd.DataFrame(np.ones((4, len(cols))), columns=cols, index=[9, 5, 2, 1])
    x["ndvi_mean"] = 0
    mask = np.array([True, False, True, True])
    ref, arm = texture_arms(x, ["svm_rbf"], mask, seed=42, mode="legacy")
    assert ref.x.shape == (4, 60)
    assert arm.x.shape == (4, 80)
    assert arm.reference == ref.name
    assert arm.x[ref.x.columns].equals(ref.x)
    assert np.array_equal(ref.eligible, arm.eligible)
    assert arm.x.index.equals(x.index)
    with pytest.raises(ValueError):
        texture_arms(x.drop(columns=cols[0]), ["svm_rbf"], mask, seed=42, mode="legacy")


def test_geographic_reference_respects_fold_labels_qa_and_ties():
    manifest = pd.DataFrame({"latitude": [0.] * 6, "longitude": [0., 2., 1., 1., 0., 2.]})
    folds = [
        SimpleNamespace(train_idx=np.array([0, 1, 2]), test_idx=np.array([3, 4, 5]),
                        y_train=np.array([0, 1, 1]), y_test=np.array([1, 0, 1]), label_threshold_ppm=20.),
        SimpleNamespace(train_idx=np.array([3, 4, 5]), test_idx=np.array([0, 1, 2]),
                        y_train=np.array([1, 0, 1]), y_test=np.array([0, 1, 1]), label_threshold_ppm=20.),
    ]
    table, scores, distance, overall = evaluate_geographic_baseline(
        manifest, folds, quality_mask=np.array([True, True, False, True, True, True]))
    assert scores[3] == .5
    assert scores[4] == 0
    assert scores[5] == 1
    assert distance[3] > 100_000
    assert table.n_train.tolist() == [2, 3]
    assert overall["n"] == 6
    with pytest.raises(ValueError):
        evaluate_geographic_baseline(manifest, folds, quality_mask=np.ones(6, dtype=int))


def test_spatial_components_join_scene_across_grid_and_buffer_preserves_test():
    from lithiumscope.model_2.validation.sensitivity import plan_protocol, spatial_components

    frame = pd.DataFrame({
        "latitude": [0.] * 8,
        "longitude": [.499, .501, 2., 2.01, 4., 4.01, 6., 6.01],
        "scene_id": ["shared", "shared", "b", "b", "c", "c", "d", "d"],
    })
    groups = spatial_components(frame, .5, combine_scene=True)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.nunique() == 4
    _, records = plan_protocol(
        frame, [10, 30] * 4, grid=.5, buffer_m=1000., combine_scene=False,
        folds=5, endpoint_ppm=20.)
    assert sorted(i for row in records for i in row["test"]) == list(range(8))
    assert sum(row["n_embargoed"] for row in records) >= 1
    for row in records:
        assert not set(row["train"]) & set(row["test"])
        assert row["minimum_fit_test_m"] >= 1000.
    with pytest.raises(ValueError):
        spatial_components(frame.drop(columns="scene_id"), .5, combine_scene=True)
