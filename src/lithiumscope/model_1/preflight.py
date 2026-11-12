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


from lithiumscope.model_1.preflight_run_model_1_preflight import run_model_1_preflight
