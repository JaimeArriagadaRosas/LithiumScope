from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.logger import get_logger
from lithiumscope.core.pipeline_audit import PipelineAudit
from lithiumscope.datasets.harmonization import canonicalize_coordinates
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.model_1.steps.step_01_load_data import load_data
from lithiumscope.model_1.steps.step_02_detection_limits import clean_detection_limits
from lithiumscope.model_1.steps.step_03_missing_values import normalize_missing_values, should_include_age
from lithiumscope.model_1.steps.step_04_quality_control import apply_quality_control
from lithiumscope.model_1.steps.step_05_target_filtering import filter_target
from lithiumscope.model_1.steps.step_06_category_cleaning import clean_categories
from lithiumscope.model_1.steps.step_07_feature_engineering import add_geochemical_features
from lithiumscope.model_1.steps.step_08_preprocessing import FeatureSchema, select_feature_schema
from lithiumscope.model_1.steps.step_09_validation import (
    build_spatial_groups,
    spatial_coordinate_diagnostics,
)

logger = get_logger("model_1.pipeline")


@dataclass
class PreparedModel1Data:
    raw_frame: pd.DataFrame
    frame: pd.DataFrame
    target: str
    schema: FeatureSchema
    x: pd.DataFrame
    y: pd.Series
    audit: PipelineAudit
    groups: pd.Series | None = None
    coordinate_report: dict = field(default_factory=dict)
    outer_splits: list[tuple[np.ndarray, np.ndarray]] | None = None
    validation_strategy: str | None = None


def prepare_training_data(path: Path, model_family: str = "common") -> PreparedModel1Data:
    config = load_config("model_1")
    data_cfg = config["data"]
    audit = PipelineAudit()

    frame = load_data(path)
    raw_frame = frame.copy()
    audit.capture("01_load_data", frame)

    frame = clean_detection_limits(frame)
    audit.capture("02_detection_limits", frame)

    frame = normalize_missing_values(frame)
    audit.capture("03_missing_values", frame)

    frame = canonicalize_coordinates(frame)
    coordinate_report = spatial_coordinate_diagnostics(
        frame
    )
    audit.capture(
        "04_canonical_coordinates",
        frame,
        (
            f"valid={coordinate_report['valid_rows']} "
            f"missing={coordinate_report['missing_rows']} "
            f"out_of_range={coordinate_report['out_of_range_rows']}"
        ),
    )

    include_age = should_include_age(
        frame,
        max_missing_fraction=float(data_cfg["age_max_missing_fraction"]),
    )

    frame, target = filter_target(
        frame,
        candidates=list(data_cfg["target_candidates"]),
        lower_quantile=float(data_cfg["target_lower_quantile"]),
        upper_quantile=float(data_cfg["target_upper_quantile"]),
    )
    audit.capture(
        "05_target_validity",
        frame,
        f"target={target}; no target-quantile trimming",
    )

    frame = apply_quality_control(
        frame,
        candidates=list(data_cfg["quality_column_candidates"]),
        minimum=float(data_cfg["quality_min"]),
        maximum=float(data_cfg["quality_max"]),
    )
    audit.capture("06_quality_control", frame)

    frame = clean_categories(
        frame,
        model_family="common",
    )
    audit.capture(
        "07_category_cleaning",
        frame,
        "common semantic rules for all algorithms",
    )

    frame = add_geochemical_features(frame)
    audit.capture("08_feature_engineering", frame)

    coordinate_report = spatial_coordinate_diagnostics(
        frame
    )
    schema = select_feature_schema(frame, include_age=include_age)
    validate_training_schema(target, schema.numeric, schema.categorical)

    for column in schema.numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    columns = schema.numeric + schema.categorical
    if not columns:
        raise ValueError("No usable predictor columns were found in the dataset.")

    x = frame[columns].copy()
    y = pd.to_numeric(frame[target], errors="coerce")
    validation_cfg = config["validation"]
    groups = None
    if bool(
        validation_cfg.get(
            "prefer_spatial_groups",
            False,
        )
    ):
        groups = build_spatial_groups(
            frame,
            degrees=float(
                validation_cfg.get(
                    "spatial_group_degrees",
                    0.5,
                )
            ),
        )

    audit.capture(
        "09_validation_inputs",
        frame,
        (
            f"features={len(columns)} "
            f"spatial_groups={groups.nunique() if groups is not None else 0}"
        ),
    )
    logger.info(
        "Prepared Model 1 matrix: X=%s y=%s spatial_groups=%s requested_family=%s",
        x.shape,
        y.shape,
        groups.nunique() if groups is not None else 0,
        model_family,
    )
    print(f"    ✓ Matriz de entrenamiento: X={x.shape}, y={y.shape}")
    if groups is not None:
        print(
            "    ✓ Grupos espaciales candidatos: "
            f"{groups.nunique()}"
        )
    return PreparedModel1Data(
        raw_frame=raw_frame,
        frame=frame,
        target=target,
        schema=schema,
        x=x,
        y=y,
        audit=audit,
        groups=groups,
        coordinate_report=coordinate_report,
    )


def prepare_prediction_data(frame: pd.DataFrame, model_family: str, schema: FeatureSchema) -> pd.DataFrame:
    result = clean_detection_limits(frame)
    result = normalize_missing_values(result)
    result = canonicalize_coordinates(result)
    result = clean_categories(result, model_family="common")
    result = add_geochemical_features(result)

    missing = [
        column
        for column in schema.numeric + schema.categorical
        if column not in result.columns
    ]
    if missing:
        logger.warning("Prediction input is missing %d expected columns: %s", len(missing), missing)

    for column in schema.numeric + schema.categorical:
        if column not in result.columns:
            result[column] = pd.NA
    for column in schema.numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[schema.numeric + schema.categorical].copy()
