from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.preprocessing")

OXIDES = [
    "SiO2", "TiO2", "Al2O3", "Fe2O3", "MnO",
    "MgO", "CaO", "Na2O", "K2O", "P2O5",
]
TRACE_ELEMENTS = [
    "Th_icpms", "U_icpms", "Rb_icpms", "Cs_icpms", "Nb_icpms", "Ta_icpms",
    "Pb_icpms", "Ba_icpms", "Sr_icpms", "Zr_icpms", "V_icpms", "Hf_icpms",
]
SPATIAL_CANDIDATES = [
    "Longitude (X)", "Logintude (X)", "Longitude", "longitude",
    "Latitude (Y)", "Latitude", "latitude",
]
CATEGORICAL = ["Geologycal_age", "Sample_type", "Rock_type", "Arc", "Domain"]
DERIVED = ["Alkali_Sum", "Mg_Number", "A_CNK_proxy", "K_Mg_ratio"]
AGE_CANDIDATES = ["Age (Ma)", "Age (ma)", "Age", "age_ma"]


@dataclass(frozen=True)
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]


def select_feature_schema(frame: pd.DataFrame, include_age: bool = True) -> FeatureSchema:
    numeric: list[str] = []
    for candidate in OXIDES + TRACE_ELEMENTS + SPATIAL_CANDIDATES + DERIVED:
        if candidate in frame.columns and candidate not in numeric:
            numeric.append(candidate)
    if include_age:
        for candidate in AGE_CANDIDATES:
            if candidate in frame.columns:
                numeric.append(candidate)
                break
    categorical = [column for column in CATEGORICAL if column in frame.columns]
    logger.info("Feature schema: %d numeric + %d categorical", len(numeric), len(categorical))
    return FeatureSchema(numeric=numeric, categorical=categorical)


def build_preprocessor(schema: FeatureSchema, model_family: str) -> ColumnTransformer:
    family = model_family.lower()

    if family == "xgboost":
        numeric_transformer = "passthrough"
    elif family == "tabnet":
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
        )
    else:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if family == "svm":
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, schema.numeric),
            ("categorical", categorical_transformer, schema.categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
