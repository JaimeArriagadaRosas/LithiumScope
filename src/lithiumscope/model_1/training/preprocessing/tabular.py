from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from lithiumscope.core.logging.session import get_logger

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
    "Longitude",
    "Latitude",
]
CATEGORICAL = ["Geologycal_age", "Sample_type", "Rock_type", "Arc", "Domain"]
DERIVED = ["Alkali_Sum", "Mg_Number", "A_CNK_proxy", "K_Mg_ratio"]
AGE_CANDIDATES = ["Age (Ma)", "Age (ma)", "Age", "age_ma"]


@dataclass(frozen=True)
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]
    preparation_version: int = 1
    preprocessing_policy: str = "legacy"


def select_feature_schema(frame: pd.DataFrame, include_age: bool = True, *, preparation_version: int = 2, preprocessing_policy: str = "legacy") -> FeatureSchema:
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
    return FeatureSchema(numeric=numeric, categorical=categorical, preparation_version=preparation_version, preprocessing_policy=preprocessing_policy)


def build_preprocessor(schema: FeatureSchema, model_family: str) -> ColumnTransformer:
    family = model_family.lower()
    policy=getattr(schema,"preprocessing_policy","legacy")
    if policy not in {"legacy","stable_v1","tabnet_mask_v1","knn_v1"}:
        raise ValueError("Unknown preprocessing policy")
    if policy=="tabnet_mask_v1" and family!="tabnet":
        raise ValueError("TabNet mask variant is only valid for TabNet")
    stable=policy!="legacy"

    if policy == "knn_v1":
        # Scale before neighbor distances; fit within each estimator TRAIN.
        numeric_transformer = Pipeline([("scale", RobustScaler()),
            ("imputer", KNNImputer(n_neighbors=5, weights="distance", keep_empty_features=True))])
    elif family == "xgboost":
        numeric_transformer = "passthrough"
    elif family == "tabnet":
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=stable))]
        )
    else:
        numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=stable))]
        if family == "svm":
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    if stable:
        from lithiumscope.model_1.training.preprocessing.encoding import ExplicitCategoryEncoder
        categorical_transformer=ExplicitCategoryEncoder()
    transformers=[
        ("numeric",numeric_transformer,schema.numeric),
        ("categorical",categorical_transformer,schema.categorical),
    ]
    if policy=="tabnet_mask_v1":
        from sklearn.impute import MissingIndicator
        transformers.append(("missing_mask",MissingIndicator(features="all"),schema.numeric))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
