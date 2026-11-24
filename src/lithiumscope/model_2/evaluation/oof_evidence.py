"""Traceable out-of-fold evidence with per-fold geochemical label thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.model_2.evaluation.metrics import validate_binary_inputs


def build_oof_table(manifest, lithium, labels, fold_ids, probabilities, scientific_folds, quality_mask=None):
    y, scores = validate_binary_inputs(labels, probabilities)
    n = len(y)
    ids = np.asarray(fold_ids, dtype=int)
    li = np.asarray(lithium, dtype=float)
    if len(manifest) != n or ids.shape != (n,) or li.shape != (n,):
        raise ValueError("Manifest, OOF scores and labels must align")
    if manifest.sample_id.isna().any() or manifest.sample_id.duplicated().any():
        raise ValueError("OOF sample identities must be present and unique")
    if not np.allclose(manifest.Li_icpms.to_numpy(dtype=float), li, rtol=0, atol=1e-10):
        raise ValueError("Manifest lithium values differ from evaluated rows")
    from lithiumscope.core.scientific_checks import assert_partition_coverage
    assert_partition_coverage([(f.train_idx, f.test_idx) for f in scientific_folds], n)
    for i, fold in enumerate(scientific_folds, 1):
        if not np.all(ids[fold.test_idx] == i) or not np.array_equal(y[fold.test_idx], fold.y_test):
            raise ValueError("OOF fold assignments or labels differ from protocol")
    thresholds = {i: fold.label_threshold_ppm for i, fold in enumerate(scientific_folds, 1)}
    threshold = pd.Series(ids).map(thresholds)
    if threshold.isna().any():
        raise ValueError("OOF fold identifier is absent from protocol")
    columns = [c for c in ("sample_id", "longitude", "latitude", "spatial_group", "scene_id") if c in manifest]
    table = manifest[columns].reset_index(drop=True).copy()
    table.insert(0, "row_position", np.arange(n))
    table["li_ppm"] = li
    table["y_true"] = y
    table["outer_fold"] = ids
    table["label_threshold_ppm"] = threshold
    table["prospectivity_score"] = scores
    table["predicted_class_0_5"] = (scores >= 0.5).astype(int)
    table["eligible_for_training"] = np.ones(n, dtype=bool) if quality_mask is None else np.asarray(quality_mask, dtype=bool)
    return table
