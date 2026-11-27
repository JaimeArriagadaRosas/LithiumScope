"""Construct and persist scientific preflight evidence without training estimators."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def feature_manifest(prepared) -> pd.DataFrame:
    categorical = set(prepared.schema.categorical)
    rows = []
    for order, column in enumerate(prepared.x.columns, start=1):
        series = prepared.x.iloc[:, order - 1]
        rows.append(
            {
                "order": order,
                "feature": str(column),
                "kind": (
                    "categorical"
                    if column in categorical
                    else "numeric"
                ),
                "missing_count": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "unique_non_null": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows, columns=["order", "feature", "kind", "missing_count", "missing_fraction", "unique_non_null"])


def source_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "source_dataset" not in frame.columns:
        return {}
    values = (
        frame["source_dataset"]
        .astype("string")
        .fillna("unknown")
    )
    return {
        str(key): int(value)
        for key, value in values.value_counts(dropna=False).items()
    }


def write_preflight_evidence(prepared, run_context, payload, feature_manifest, outer_splits):
    manifest_path = (
        run_context.manifests
        / "model_1_preflight.json"
    )
    manifest_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    feature_manifest.to_csv(
        run_context.tables
        / "model_1_feature_manifest.csv",
        index=False,
    )
    if outer_splits is not None:
        fold_assignment = np.full(
            len(prepared.y),
            -1,
            dtype=int,
        )
        for fold, (_, test_idx) in enumerate(
            outer_splits,
            start=1,
        ):
            fold_assignment[test_idx] = fold
        fold_manifest = pd.DataFrame(
            {
                "row_position": np.arange(len(prepared.y)),
                "row_index": prepared.y.index.astype(str),
                "outer_fold": fold_assignment,
            }
        )
        if "source_dataset" in prepared.frame.columns:
            fold_manifest["source_dataset"] = (
                prepared.frame["source_dataset"].astype("string").to_numpy()
            )
        if prepared.groups is not None:
            fold_manifest["spatial_group"] = (
                prepared.groups.astype("string").to_numpy()
            )
        for coordinate_name in ("Longitude", "Latitude"):
            if coordinate_name in prepared.frame.columns:
                fold_manifest[coordinate_name] = pd.to_numeric(
                    prepared.frame[coordinate_name],
                    errors="coerce",
                ).to_numpy()
        fold_manifest.to_csv(
            run_context.tables
            / "outer_fold_manifest.csv",
            index=False,
        )
