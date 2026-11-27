import numpy as np
import pandas as pd
import pytest
from lithiumscope.core.scientific.spatial_audit import audit_spatial_folds


def test_separate_cells_may_have_overlapping_patches_and_scenes():
    frame = pd.DataFrame({"latitude": [0, 0, 0, 0], "longitude": [0, .001, .002, .003],
                          "sample_id": ["a","b","c","d"], "sentinel_scene_id": ["s","s","s","t"]})
    groups = pd.Series(["g1","g2","g2","g1"])
    splits = [(np.array([0,3]), np.array([1,2])), (np.array([1,2]), np.array([0,3]))]
    summary, details = audit_spatial_folds(frame, groups, splits, patch_side_m=640,
                                          minimum_distance_m=500)
    assert len(details) == 4
    assert summary.possible_patch_overlap_test_rows.tolist() == [2,2]
    assert not summary.distance_policy_pass.any()
    assert summary.sentinel_scene_id_shared_test_rows.tolist() == [2,1]
    assert summary.minimum_fit_m.iloc[0] == pytest.approx(111.195, rel=1e-3)


def test_quality_mask_excludes_only_fit_distances():
    frame = pd.DataFrame({"latitude": [0,0,0,0], "longitude": [0, .1, .001, .2]})
    groups = pd.Series(["a","a","b","b"])
    splits = [(np.array([0,1]), np.array([2,3])), (np.array([2,3]), np.array([0,1]))]
    summary, details = audit_spatial_folds(frame, groups, splits, eligible=[False,True,True,True])
    assert len(details) == 4
    assert summary.fit_rows.tolist() == [1,2]
    assert summary.minimum_fit_m.iloc[0] > summary.minimum_all_train_m.iloc[0]
    with pytest.raises(ValueError, match="coverage"):
        audit_spatial_folds(frame, groups, [(np.array([0]), np.array([2,3]))])
