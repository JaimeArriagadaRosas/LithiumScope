import json
from pathlib import Path

import pandas as pd
import pytest

from lithiumscope.core.scientific_checks import ScientificValidationError
from lithiumscope.datasets.harmonization import canonicalize_coordinates
from lithiumscope.model_1.steps.step_05_target_filtering import filter_target
from lithiumscope.model_1.steps.step_06_category_cleaning import clean_categories
from lithiumscope.model_1.steps.step_09_validation import materialize_regression_splits


def test_coordinates_are_coalesced_row_by_row():
    frame = pd.DataFrame(
        {
            "Longitude": [-70.0, None],
            "Longitude (X)": [None, -69.0],
            "Latitude": [-33.0, None],
            "Latitude (Y)": [None, -22.0],
        }
    )

    result = canonicalize_coordinates(frame)

    assert result["Longitude"].tolist() == [-70.0, -69.0]
    assert result["Latitude"].tolist() == [-33.0, -22.0]


def test_target_quantiles_are_diagnostic_not_row_filters():
    frame = pd.DataFrame(
        {
            "Li_icpms": list(range(1, 101)),
        }
    )

    result, target = filter_target(
        frame,
        ["Li_icpms"],
        0.025,
        0.975,
    )

    assert target == "Li_icpms"
    assert len(result) == 100
    assert result[target].min() == 1
    assert result[target].max() == 100


def test_required_spatial_validation_never_falls_back_silently():
    x = pd.DataFrame({"feature": range(10)})
    y = pd.Series(range(10), dtype=float)

    with pytest.raises(
        ScientificValidationError,
        match="Random KFold fallback is disabled",
    ):
        materialize_regression_splits(
            x,
            y,
            groups=None,
            folds=5,
            seed=42,
            require_groups=True,
        )


def test_semantic_category_cleaning_is_algorithm_independent():
    frame = pd.DataFrame(
        {
            "Rock_type": ["granodiorite", "serpentinite"],
            "Sample_type": ["pyroclastic", "metamorphic"],
        }
    )

    random_forest = clean_categories(
        frame,
        "random_forest",
    )
    svm = clean_categories(
        frame,
        "svm",
    )

    pd.testing.assert_frame_equal(
        random_forest,
        svm,
    )
