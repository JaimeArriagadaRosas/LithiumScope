"""Supplementary geographic reference on the exact outer scientific folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from lithiumscope.core.scientific.spatial_audit import geographic_coordinates, EARTH_RADIUS_M
from lithiumscope.core.scientific.classification_metrics import classification_metrics
from lithiumscope.model_2.validation.spatial import scientific_labels


def evaluate_geographic_baseline(manifest, folds, *, quality_mask=None):
    """Nearest eligible TRAIN observation, averaging exact-distance ties.

    Coordinates are available at georeferenced inference. This diagnostic
    is not a spectral candidate, calibrated model, or promotion criterion.
    """
    n = len(manifest)
    labels = scientific_labels(folds, n)
    coords = geographic_coordinates(manifest)
    mask = np.ones(n, dtype=bool) if quality_mask is None else np.asarray(quality_mask)
    if mask.shape != (n,) or mask.dtype.kind != "b":
        raise ValueError("Geographic baseline QA mask must align")
    scores = np.full(n, np.nan)
    distances = np.full(n, np.nan)
    records = []
    for number, fold in enumerate(folds, 1):
        eligible = mask[fold.train_idx]
        train = fold.train_idx[eligible]
        targets = np.asarray(fold.y_train)[eligible]
        if set(np.unique(targets)) != {0, 1}:
            raise ValueError("Geographic baseline requires both eligible TRAIN classes")
        tree = BallTree(coords[train], metric="haversine")
        test = fold.test_idx
        nearest, _ = tree.query(coords[test], k=1)
        # Account for numerical roundoff at the minimum; do not resolve ties by row order.
        ties = tree.query_radius(coords[test], r=nearest[:, 0] + 1e-12)
        prediction = np.array([targets[index].mean() for index in ties], dtype=float)
        scores[test] = prediction
        distances[test] = nearest[:, 0] * EARTH_RADIUS_M
        records.append({"fold": number, "n_train": len(train), "n_test": len(test),
                        "label_threshold_ppm": fold.label_threshold_ppm,
                        **classification_metrics(fold.y_test, prediction)})
    overall = classification_metrics(labels, scores)
    return pd.DataFrame(records), scores, distances, overall
