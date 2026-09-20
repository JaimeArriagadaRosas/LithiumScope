import pandas as pd

from lithiumscope.prediction.analysis import (
    concordance_table,
    correlation_table,
    pair_model_outputs,
)
from lithiumscope.prediction.interpretation import integrated_case_text


def _paired_inputs():
    m1 = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "d"],
            "Li_icpms": [30.0, 5.0, 25.0, 7.0],
            "Li_icpms_predicted": [28.0, 6.0, 20.0, 9.0],
            "out_of_training_range_fraction": [0.1, 0.1, 0.1, 0.1],
        }
    )
    m2 = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "d"],
            "prospectivity_score": [0.8, 0.2, 0.7, 0.1],
            "priority": ["alta", "baja", "alta", "baja"],
            "out_of_training_range_fraction": [0.1, 0.1, 0.1, 0.1],
        }
    )
    return m1, m2


def test_pairing_and_correlations_use_case_id():
    m1, m2 = _paired_inputs()
    paired = pair_model_outputs(m1, m2)
    correlations = correlation_table(paired)

    assert len(paired) == 4
    assert not correlations.empty
    assert "Li predicho M1 ↔ score M2" in set(correlations["relationship"])


def test_concordance_is_post_prediction_not_model_input():
    m1, m2 = _paired_inputs()
    paired = pair_model_outputs(m1, m2)
    table = concordance_table(
        paired,
        lithium_threshold_ppm=21.6,
        model_2_probability_threshold=0.5,
    )

    assert table.loc[table["case_id"] == "a", "concordance"].item() == "concordante_alta"
    assert table.loc[table["case_id"] == "b", "concordance"].item() == "concordante_baja"
    assert table.loc[table["case_id"] == "c", "concordance"].item() == "divergente_m2_alto"


def test_no_fake_correlation_without_paired_cases():
    m1, m2 = _paired_inputs()
    m2["case_id"] = ["x", "y", "z", "w"]
    paired = pair_model_outputs(m1, m2)

    assert paired.empty
    assert correlation_table(paired).empty


def test_integrated_interpretation_distinguishes_divergence():
    row = pd.Series(
        {
            "concordance": "divergente_m1_alto",
            "model_1_out_of_training_range_fraction": 0.1,
            "model_2_out_of_training_range_fraction": 0.1,
        }
    )
    text = integrated_case_text(row)

    assert "divergen" in text.lower()
    assert "error" not in text.lower()
