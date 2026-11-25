import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from lithiumscope.core.scientific.nested_protocol import audit_nested_development, nested_summary


def cohort():
    groups = pd.Series(np.repeat(np.arange(12), 4).astype(str))
    target = pd.Series(np.tile([1., 5., 10., 20.], 12))
    frame = pd.DataFrame({"latitude": np.repeat(np.arange(12), 4),
                          "longitude": np.zeros(48)})
    splits = list(GroupKFold(3).split(frame, target, groups))
    return frame, groups, target, splits


def test_nested_regression_uses_only_outer_train_and_final_development():
    frame, g, y, splits = cohort()
    table = audit_nested_development(frame, g, y, splits, inner_folds=2)
    assert len(table) == 8
    assert nested_summary(table)["blocked"] == 0
    assert set(table.loc[table.outer_fold > 0, "parent_rows"]) == {32}
    assert set(table.loc[table.outer_fold == 0, "parent_rows"]) == {48}
    assert table.minimum_fit_m.min() > 100000


def test_nested_classification_distinguishes_endpoints_and_respects_qa():
    frame, g, y, splits = cohort()
    keep = np.ones(len(y), dtype=bool)
    keep[0] = False
    table = audit_nested_development(frame, g, y, splits, inner_folds=2,
        eligible=keep, endpoints=[9.]*3, final_endpoint=9., quantile=.75,
        operating=True, calibration=True)
    assert nested_summary(table)["state"] == "ready"
    assert set(table.loc[table.phase == "optimization", "label_threshold_ppm"]) != {9.}
    assert set(table.loc[table.phase != "optimization", "label_threshold_ppm"]) == {9.}
    assert table.loc[(table.outer_fold == 0) & (table.phase == "operating"), "parent_rows"].eq(47).all()
    assert table.phase.str.startswith("operating_calibration").any()


def test_nested_preflight_blocks_single_class_and_physical_overlap():
    frame, g, y, splits = cohort()
    table = audit_nested_development(frame, g, y, splits, inner_folds=2,
        endpoints=[100.]*3, final_endpoint=100., operating=True,
        optimization=False, minimum_distance_m=0)
    assert nested_summary(table)["blocked"] == len(table)
    assert set(table.reason) == {"single_class"}
    table = audit_nested_development(frame, g, y, splits, inner_folds=2,
                                    minimum_distance_m=1e7)
    assert nested_summary(table)["blocked"] == len(table)
    with pytest.raises(ValueError, match="boolean"):
        audit_nested_development(frame, g, y, splits, inner_folds=2, eligible=np.ones(len(y), int))
