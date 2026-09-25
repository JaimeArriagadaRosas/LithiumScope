import pandas as pd

from lithiumscope.model_1.steps.step_06_category_cleaning import (
    clean_categories,
)


def test_category_cleaning_preserves_canonical_georoc_materials():
    frame = pd.DataFrame(
        {
            "Sample_type": [
                "WHOLE ROCK",
                "VOLCANIC GLASS",
            ]
        }
    )

    cleaned = clean_categories(
        frame,
        "random_forest",
    )

    assert cleaned["Sample_type"].tolist() == [
        "whole_rock",
        "volcanic_glass",
    ]


def test_category_cleaning_preserves_georoc_codes_if_present():
    frame = pd.DataFrame(
        {
            "Sample_type": [
                "WR",
                "GL",
            ]
        }
    )

    cleaned = clean_categories(
        frame,
        "svm",
    )

    assert cleaned["Sample_type"].tolist() == [
        "whole_rock",
        "volcanic_glass",
    ]
