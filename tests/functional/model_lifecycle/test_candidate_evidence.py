import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.core.scientific.candidate_evidence import eligible_rows, record_evidence, verify_candidate
from lithiumscope.core.scientific_checks import assert_partition_coverage
from lithiumscope.model_2.evaluation.oof_evidence import build_oof_table


def fail():
    raise OSError("simulated export failure")


def test_optional_failure_does_not_disqualify_valid_candidate():
    row = {"status": "ok"}
    tracker = SimpleNamespace(event=lambda *a, **kw: None)
    record_evidence(row, tracker, logging.getLogger(__name__), "rf",
                    {"tables": lambda: None}, {"plot": fail})
    assert row["status"] == "ok" and row["artifact_state"] == "partial"
    assert eligible_rows(pd.DataFrame([row])).iloc[0]


def test_required_failure_disqualifies_without_duplicate_training_row():
    row = {"status": "ok"}
    tracker = SimpleNamespace(event=lambda *a, **kw: None)
    record_evidence(row, tracker, logging.getLogger(__name__), "rf",
                    {"tables": fail}, {})
    assert row["status"] == "ok" and row["artifact_state"] == "failed"
    assert not eligible_rows(pd.DataFrame([row])).iloc[0]


def test_invalid_candidate_metrics_rejected():
    result = SimpleNamespace(fold_table=pd.DataFrame({"fold": [1, 2], "rmse": [1, np.nan]}),
                             overall_metrics={"rmse": 1})
    with pytest.raises(ValueError):
        verify_candidate(result, [1, 2], 2, ("rmse",), 2)


def test_outer_partition_requires_exact_coverage():
    assert_partition_coverage([(np.array([1]), np.array([0])),
                               (np.array([0]), np.array([1]))], 2)
    with pytest.raises(ValueError):
        assert_partition_coverage([(np.array([1]), np.array([0, 0]))], 2)
    with pytest.raises(ValueError):
        assert_partition_coverage([(np.array([1]), np.array([0]))], 2)


def test_oof_identity_preserves_ineligible_test_rows():
    manifest = pd.DataFrame({"sample_id": ["a", "b", "c", "d"],
                             "Li_icpms": [1, 3, 2, 4]})
    folds = [SimpleNamespace(train_idx=np.array([2, 3]), test_idx=np.array([0, 1]),
                             y_test=np.array([0, 1]), label_threshold_ppm=2.5),
             SimpleNamespace(train_idx=np.array([0, 1]), test_idx=np.array([2, 3]),
                             y_test=np.array([0, 1]), label_threshold_ppm=3)]
    table = build_oof_table(manifest, [1, 3, 2, 4], [0, 1, 0, 1], [1, 1, 2, 2],
                            [.2, .8, .3, .7], folds, [True, False, True, True])
    assert table.sample_id.tolist() == ["a", "b", "c", "d"]
    assert table.label_threshold_ppm.tolist() == [2.5, 2.5, 3, 3]
    assert not table.eligible_for_training.iloc[1]
    with pytest.raises(ValueError):
        build_oof_table(manifest, [1, 3, 2, 4], [0, 1, 0, 1], [2, 1, 2, 2],
                        [.2, .8, .3, .7], folds)
