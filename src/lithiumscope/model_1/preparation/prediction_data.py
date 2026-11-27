"""Deterministic inference preparation using the frozen feature schema."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.logging.session import get_logger
from lithiumscope.datasets.harmonization import canonicalize_coordinates
from lithiumscope.datasets.validation.spatial.coordinates import assess_coordinate_aliases, coordinate_summary
from lithiumscope.datasets.validation.tabular.missing_values import normalize_missing_values
from lithiumscope.model_1.preparation.category_cleaning import clean_categories
from lithiumscope.model_1.preparation.detection_limits import clean_detection_limits
from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
from lithiumscope.model_1.preparation.missing_values import normalize_numeric_predictors
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema

logger = get_logger("model_1.preparation.prediction")


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
    result = normalize_numeric_predictors(result, schema)
    output = result[schema.numeric + schema.categorical].copy()
    output.attrs["prediction_provenance"] = {"missing_expected_columns": missing,
        "generated_during_preparation": generated}
    if coordinate_evidence is not None:
        output.attrs["prediction_provenance"]["coordinate_status"]=coordinate_evidence.coordinate_status.tolist()
        output.attrs["prediction_provenance"]["coordinate_report"]=coordinate_summary(coordinate_evidence)
    return output
