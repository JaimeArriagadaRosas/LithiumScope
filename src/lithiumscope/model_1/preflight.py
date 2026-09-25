from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lithiumscope.core.logger import get_logger
from lithiumscope.core.scientific_checks import (
    ScientificValidationError,
    assert_group_isolation,
    assert_no_target_leakage,
)
from lithiumscope.model_1.steps.step_09_validation import (
    materialize_regression_splits,
)

logger = get_logger("model_1.preflight")

_IDENTIFIER_COLUMNS = {
    "source_dataset",
    "source_sample",
    "sample_id",
    "sample",
    "unique_id",
    "id",
}


def _feature_manifest(prepared) -> pd.DataFrame:
    categorical = set(prepared.schema.categorical)
    rows = []
    for order, column in enumerate(prepared.x.columns, start=1):
        series = prepared.x[column]
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
    return pd.DataFrame(rows)


def _source_counts(frame: pd.DataFrame) -> dict[str, int]:
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


def run_model_1_preflight(
    prepared,
    config: dict,
    run_context,
) -> dict:
    validation = config["validation"]
    prefer_spatial = bool(
        validation.get("prefer_spatial_groups", False)
    )
    allow_random = bool(
        validation.get("allow_random_fallback", False)
    )
    require_groups = prefer_spatial and not allow_random
    outer_folds = int(validation["outer_folds"])
    seed = int(validation["random_seed"])

    errors: list[str] = []
    warnings: list[str] = []
    features = [str(column) for column in prepared.x.columns]

    try:
        assert_no_target_leakage(
            prepared.target,
            features,
        )
    except ScientificValidationError as exc:
        errors.append(str(exc))

    normalized = {
        feature.strip().lower()
        for feature in features
    }
    leaked_identifiers = sorted(
        normalized.intersection(_IDENTIFIER_COLUMNS)
    )
    if leaked_identifiers:
        errors.append(
            "Identifier/provenance columns are present in predictors: "
            + ", ".join(leaked_identifiers)
        )

    duplicated_feature_names = [
        str(name)
        for name in prepared.x.columns[
            prepared.x.columns.duplicated()
        ]
    ]
    if duplicated_feature_names:
        errors.append(
            "Duplicate predictor columns: "
            + ", ".join(duplicated_feature_names)
        )

    target_values = pd.to_numeric(
        prepared.y,
        errors="coerce",
    ).to_numpy(dtype=float)
    non_finite_target = int(
        (~np.isfinite(target_values)).sum()
    )
    if non_finite_target:
        errors.append(
            f"Target contains {non_finite_target} non-finite values."
        )

    feature_manifest = _feature_manifest(prepared)
    all_missing = feature_manifest.loc[
        feature_manifest["missing_fraction"] >= 1.0,
        "feature",
    ].astype(str).tolist()
    if all_missing:
        errors.append(
            "Predictors entirely missing: "
            + ", ".join(all_missing)
        )

    constant = feature_manifest.loc[
        feature_manifest["unique_non_null"] <= 1,
        "feature",
    ].astype(str).tolist()
    if constant:
        warnings.append(
            "Constant/nearly empty predictors detected: "
            + ", ".join(constant)
        )

    coordinate = dict(prepared.coordinate_report)
    spatial_groups = (
        int(prepared.groups.nunique())
        if prepared.groups is not None
        else 0
    )
    if require_groups:
        if prepared.groups is None:
            errors.append(
                "Spatial validation is mandatory but complete valid coordinates "
                "are not available for every retained sample. "
                f"valid={coordinate.get('valid_rows', 0)} "
                f"missing={coordinate.get('missing_rows', 0)} "
                f"out_of_range={coordinate.get('out_of_range_rows', 0)}."
            )
        elif spatial_groups < outer_folds:
            errors.append(
                "Spatial validation is mandatory but only "
                f"{spatial_groups} groups are available for {outer_folds} folds."
            )

    exact_duplicates = int(
        prepared.frame[
            [
                *prepared.schema.numeric,
                *prepared.schema.categorical,
                prepared.target,
            ]
        ].duplicated().sum()
    )
    if exact_duplicates:
        warnings.append(
            f"{exact_duplicates} exact predictor+target duplicate rows remain."
        )

    coordinate_duplicates = 0
    if {"Longitude", "Latitude"} <= set(prepared.frame.columns):
        coordinate_duplicates = int(
            prepared.frame[
                ["Longitude", "Latitude"]
            ].duplicated().sum()
        )
        if coordinate_duplicates:
            warnings.append(
                f"{coordinate_duplicates} repeated coordinate pairs remain; "
                "spatial GroupKFold keeps identical cells together."
            )

    outer_splits = None
    strategy = None
    if not errors:
        outer_splits, strategy = materialize_regression_splits(
            prepared.x,
            prepared.y,
            prepared.groups if prefer_spatial else None,
            outer_folds,
            seed,
            require_groups=require_groups,
        )
        if (
            prepared.groups is not None
            and strategy == "group_kfold_spatial"
        ):
            for train_idx, test_idx in outer_splits:
                assert_group_isolation(
                    train_idx,
                    test_idx,
                    prepared.groups,
                )

    status = "pass" if not errors else "fail"
    payload = {
        "schema_version": 1,
        "status": status,
        "samples": int(len(prepared.y)),
        "target": str(prepared.target),
        "feature_count": int(len(features)),
        "numeric_features": list(prepared.schema.numeric),
        "categorical_features": list(prepared.schema.categorical),
        "source_counts": _source_counts(prepared.frame),
        "coordinate_quality": coordinate,
        "spatial_groups": spatial_groups,
        "validation": {
            "prefer_spatial_groups": prefer_spatial,
            "allow_random_fallback": allow_random,
            "outer_folds": outer_folds,
            "inner_folds": int(validation["inner_folds"]),
            "random_seed": seed,
            "strategy": strategy,
        },
        "duplicates": {
            "exact_predictor_target_rows": exact_duplicates,
            "repeated_coordinate_pairs": coordinate_duplicates,
        },
        "warnings": warnings,
        "errors": errors,
    }

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
        prepared.outer_splits = outer_splits
        prepared.validation_strategy = strategy

    run_context.tracker.event(
        "model_1_preflight_completed",
        status=status,
        errors=errors,
        warnings=warnings,
        spatial_groups=spatial_groups,
        coordinate_quality=coordinate,
        validation_strategy=strategy,
    )
    run_context.tracker.attach_summary(
        preflight_status=status,
        spatial_groups=spatial_groups,
        validation_strategy=strategy,
        coordinate_valid_rows=coordinate.get("valid_rows"),
        coordinate_missing_rows=coordinate.get("missing_rows"),
        coordinate_out_of_range_rows=coordinate.get("out_of_range_rows"),
    )

    logger.info(
        "Model 1 preflight status=%s samples=%d features=%d "
        "spatial_groups=%d strategy=%s warnings=%d errors=%d",
        status,
        len(prepared.y),
        len(features),
        spatial_groups,
        strategy,
        len(warnings),
        len(errors),
    )
    for warning in warnings:
        logger.warning("M1_PREFLIGHT_WARNING %s", warning)
    for error in errors:
        logger.error("M1_PREFLIGHT_ERROR %s", error)

    if errors:
        raise ScientificValidationError(
            "Model 1 scientific preflight failed: "
            + " | ".join(errors)
        )
    return payload
