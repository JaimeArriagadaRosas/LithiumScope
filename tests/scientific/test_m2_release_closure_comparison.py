"""Guardrails for the definitive M2 comparison against active release."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.methodology.release_closure_comparison import (
    LEGACY_OOF, NEW_OOF, RELEASE_ID, metric_summary, read_paired,
)


def _create_synthetic_files(tmp_path, *, mismatch=False, release=RELEASE_ID):
    registry = tmp_path / "models" / "active_models.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"models": {"model_2": {
            "model_path": f"models/model_2/trained/{release}/model.joblib"
        }}}), encoding="utf-8"
    )
    previous = pd.DataFrame({
        "arm": ["legacy_archived"] * 706,
        "test_index": range(706),
        "fold": np.arange(706) % 5 + 1,
        "sample_id": [f"m-{i}" for i in range(706)],
        "spatial_group": [f"g-{i % 8}" for i in range(706)],
        "y_true": np.arange(706) % 3 == 0,
        "score": np.linspace(0.1, 0.9, 706),
    })
    candidate = previous.rename(columns={"test_index": "sample_index"}).copy()
    candidate["arm"] = "control_regularized_126"
    if mismatch:
        candidate.loc[0, "y_true"] = not bool(candidate.loc[0, "y_true"])
    for path, frame in (
        (tmp_path / LEGACY_OOF, previous),
        (tmp_path / NEW_OOF, candidate),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)


def test_same_706_rows_are_accepted(tmp_path):
    _create_synthetic_files(tmp_path)
    old, new = read_paired(tmp_path)
    assert len(old) == 706 and len(new) == 706
    assert np.array_equal(old.y_true, new.y_true)


def test_any_changed_label_blocks_release_comparison(tmp_path):
    _create_synthetic_files(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="y_true"):
        read_paired(tmp_path)


def test_wrong_active_release_blocks_comparison(tmp_path):
    _create_synthetic_files(tmp_path, release="unrelated")
    with pytest.raises(ValueError, match="Active M2 release changed"):
        read_paired(tmp_path)


def test_metrics_reject_invalid_scores():
    values = pd.DataFrame({"y_true": [0, 1], "score": [0.1, np.nan]})
    with pytest.raises(ValueError, match="Invalid"):
        metric_summary(values)
