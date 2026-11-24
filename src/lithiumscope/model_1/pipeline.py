from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.pipeline_audit import PipelineAudit, stable_source_row_id
from lithiumscope.datasets.harmonization import canonicalize_coordinates
from lithiumscope.datasets.validation.coordinates import assess_coordinate_aliases, coordinate_summary
from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.datasets.readers.tabular import load_data
from lithiumscope.model_1.preparation.detection_limits import clean_detection_limits
from lithiumscope.model_1.preparation.missing_values import find_age_column, normalize_missing_values, should_include_age
from lithiumscope.model_1.preparation.quality_control import apply_quality_control
from lithiumscope.model_1.preparation.target_filtering import filter_target
from lithiumscope.model_1.preparation.category_cleaning import clean_categories
from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema, select_feature_schema, DERIVED
from lithiumscope.model_1.validation.spatial import (
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
    coordinate_evidence: pd.DataFrame | None = None


def prepare_training_data(path: Path, model_family: str = "common") -> PreparedModel1Data:
    config = load_config("model_1")
    data_cfg = config["data"]
    audit = PipelineAudit(dataset_sha256=file_sha256(path))

    frame = load_data(path)
    raw_frame = frame.copy()
    audit.capture("01_load_data", frame)

    censoring = []
    version = int(data_cfg.get("preparation_version", 2))
    frame = clean_detection_limits(frame, version=version, evidence=censoring)
    audit.censoring = censoring
    audit.capture("02_detection_limits", frame)

    frame = normalize_missing_values(frame)
    audit.capture("03_missing_values", frame)

    coordinate_evidence=None
    if version>=2:
        frame,coordinate_evidence=assess_coordinate_aliases(frame,original=raw_frame)
        coordinate_evidence["source_row_id"]=[stable_source_row_id(audit.dataset_sha256,idx) for idx in coordinate_evidence.index]
    else:
        frame=canonicalize_coordinates(frame)
    coordinate_report=spatial_coordinate_diagnostics(frame)
    if coordinate_evidence is not None:
        coordinate_report.update(coordinate_summary(coordinate_evidence))
    audit.capture(
        "04_canonical_coordinates",
        frame,
        (
            f"valid={coordinate_report['valid_rows']} "
            f"missing={coordinate_report['missing_rows']} "
            f"out_of_range={coordinate_report['out_of_range_rows']}"
        ),
        allowed_added_columns=("Longitude","Latitude"),
    )

    frame, target = filter_target(
        frame,
        candidates=list(data_cfg["target_candidates"]),
        lower_quantile=float(data_cfg["target_lower_quantile"]),
        upper_quantile=float(data_cfg["target_upper_quantile"]),
        zero_policy=str(data_cfg.get("target_zero_policy", "retain")),
    )
    audit.capture(
        "05_target_validity",
        frame,
        f"target={target}; no target-quantile trimming; {frame.attrs.get('target_validation', {})}",
        exclusion_reasons=frame.attrs.get("target_exclusions",[]),
    )

    frame = apply_quality_control(
        frame,
        candidates=list(data_cfg["quality_column_candidates"]),
        minimum=float(data_cfg["quality_min"]),
        maximum=float(data_cfg["quality_max"]),
    )
    audit.capture("06_quality_control", frame, exclusion_reasons=frame.attrs.get("qc_exclusions",[]))
    # Diagnostic coverage is not allowed to decide the predictor schema from TEST.
    include_age = bool(data_cfg.get("include_age", False))
    age_column = find_age_column(frame)
    if age_column is not None:
        age = pd.to_numeric(frame[age_column], errors="coerce")
        frame[age_column] = age.where(np.isfinite(age) & (age >= 0))
    age_coverage = should_include_age(frame, float(data_cfg["age_max_missing_fraction"]))
    if include_age and age_column is None:
        raise ValueError("The fixed training schema requires an age column.")
    logger.info("Age schema fixed include=%s; descriptive coverage>=minimum=%s", include_age, age_coverage)

    frame = clean_categories(
        frame,
        model_family="common", version=version,
    )
    audit.capture(
        "07_category_cleaning",
        frame,
        "common semantic rules for all algorithms",
    )

    frame = add_geochemical_features(frame, version=version)
    audit.capture("08_feature_engineering", frame, allowed_added_columns=DERIVED)

    coordinate_report = spatial_coordinate_diagnostics(frame)
    if coordinate_evidence is not None:
        coordinate_evidence["retained_after_qc"]=coordinate_evidence.index.isin(frame.index)
        coordinate_report.update(coordinate_summary(coordinate_evidence.loc[frame.index]))
    schema = select_feature_schema(frame, include_age=include_age, preparation_version=version, preprocessing_policy=str(data_cfg.get("preprocessing_policy","legacy")))
    validate_training_schema(target, schema.numeric, schema.categorical)

    frame = _normalize_numeric_predictors(frame, schema)

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

    if not x.index.equals(y.index) or (groups is not None and not x.index.equals(groups.index)):
        raise ValueError("Prepared X/y/groups row identity differs")
    if file_sha256(path)!=audit.dataset_sha256:
        raise ValueError("Dataset changed during preparation")
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
        coordinate_evidence=coordinate_evidence,
    )


def _normalize_numeric_predictors(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    result = frame.copy()
    version = getattr(schema, "preparation_version", 1)
    for name in schema.numeric:
        values = pd.to_numeric(result[name], errors="coerce")
        if version >= 2:
            valid = np.isfinite(values)
            if name == "Longitude":
                valid &= values.between(-180, 180)
            elif name == "Latitude":
                valid &= values.between(-90, 90)
            else:
                valid &= values.ge(0)
            values = values.where(valid)
        result[name] = values
    return result


def prepare_prediction_data(frame: pd.DataFrame, model_family: str, schema: FeatureSchema) -> pd.DataFrame:
    version = getattr(schema, "preparation_version", 1)
    result = clean_detection_limits(frame, version=version)
    result = normalize_missing_values(result)
    coordinate_evidence=None
    if version>=2:
        result,coordinate_evidence=assess_coordinate_aliases(result,original=frame)
    else:
        result=canonicalize_coordinates(result)
    result = clean_categories(result, model_family="common", version=version)
    result = add_geochemical_features(result, version=version)

    missing = [
        column
        for column in schema.numeric + schema.categorical
        if column not in result.columns
    ]
    if missing:
        logger.warning("Prediction input is missing %d expected columns: %s", len(missing), missing)

    generated = [name for name in schema.numeric + schema.categorical
                 if name not in frame.columns and name in result.columns and result[name].notna().any()]
    for column in schema.numeric + schema.categorical:
        if column not in result.columns:
            result[column] = np.nan
    result = _normalize_numeric_predictors(result, schema)
    output = result[schema.numeric + schema.categorical].copy()
    output.attrs["prediction_provenance"] = {"missing_expected_columns": missing,
        "generated_during_preparation": generated}
    if coordinate_evidence is not None:
        output.attrs["prediction_provenance"]["coordinate_status"]=coordinate_evidence.coordinate_status.tolist()
        output.attrs["prediction_provenance"]["coordinate_report"]=coordinate_summary(coordinate_evidence)
    return output
