from pathlib import Path

import pandas as pd

from lithiumscope.prediction.report import write_pdf_report


def test_pdf_report_is_generated_with_case_sections(tmp_path: Path):
    m1 = pd.DataFrame(
        [
            {
                "case_id": "case_a",
                "Li_icpms": 25.0,
                "Li_icpms_predicted": 23.0,
                "Li_icpms_interval_low_q90": 18.0,
                "Li_icpms_interval_high_q90": 28.0,
                "applicability_warning": "OK",
                "scientific_interpretation": "Interpretacion M1.",
            }
        ]
    )
    m2 = pd.DataFrame(
        [
            {
                "case_id": "case_a",
                "Li_icpms": 25.0,
                "prospectivity_score": 0.62,
                "priority": "media",
                "applicability_warning": "OK",
                "sentinel_scene_id": "scene_a",
                "scientific_interpretation": "Interpretacion M2.",
            }
        ]
    )
    paired = m1[["case_id", "Li_icpms", "Li_icpms_predicted"]].merge(
        m2[["case_id", "prospectivity_score", "priority"]],
        on="case_id",
    )
    concordance = paired.copy()
    concordance["concordance"] = "divergente_m1_alto"
    concordance["scientific_interpretation"] = "Lectura integrada."

    destination = write_pdf_report(
        tmp_path / "report.pdf",
        title="LithiumScope - Demo",
        run_id="demonstration_test",
        mode="demonstration",
        interpretation="Resumen cientifico.",
        model_1_identity={
            "algorithm": "catboost",
            "run_id": "training_a",
            "model_sha256": "a" * 64,
        },
        model_2_identity={
            "algorithm": "random_forest",
            "run_id": "training_b",
            "model_sha256": "b" * 64,
        },
        model_1_metrics={"rmse": 5.0, "mae": 4.0, "r2": 0.4},
        model_2_metrics={"roc_auc": 0.6, "average_precision": 0.5},
        model_1_predictions=m1,
        model_2_predictions=m2,
        paired=paired,
        correlations=pd.DataFrame(
            [{"relationship": "Li real <-> score M2", "n": 1, "pearson": None, "spearman": None}]
        ),
        concordance=concordance,
        training_vs_external=pd.DataFrame(),
        overlap_audit={"status": "sin coincidencias", "sample_id_matches": 0, "coordinate_matches": 0},
        figures=[],
    )

    payload = destination.read_bytes()
    assert payload.startswith(b"%PDF")
    assert len(payload) > 1000
