import pandas as pd

from lithiumscope.prediction.analysis import (
    concordance_table,
    correlation_table,
    pair_model_outputs,
)
from lithiumscope.prediction.interpretation import (
    build_overall_interpretation,
    integrated_case_text,
    model_2_case_text,
)


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


def test_model_2_human_output_changes_with_priority():
    high = model_2_case_text(
        pd.Series(
            {
                "prospectivity_score": 0.82,
                "priority": "alta",
                "out_of_training_range_fraction": 0.1,
            }
        )
    )
    low = model_2_case_text(
        pd.Series(
            {
                "prospectivity_score": 0.18,
                "priority": "baja",
                "out_of_training_range_fraction": 0.1,
            }
        )
    )

    assert "semejanza relativamente alta" in high
    assert "semejanza relativamente baja" in low
    assert high != low


def test_external_demo_interpretation_warns_small_sample():
    text = build_overall_interpretation(
        model_1_metrics={"rmse": 5.0, "mae": 4.0, "r2": 0.4},
        model_2_metrics={
            "roc_auc": 0.6,
            "average_precision": 0.4,
            "balanced_accuracy": 0.55,
        },
        correlations=pd.DataFrame(),
        concordance=pd.DataFrame(),
        model_1_case_count=10,
        model_2_case_count=10,
        paired_case_count=10,
    )

    assert "pocos casos" in text
    assert "no como validación definitiva" in text
