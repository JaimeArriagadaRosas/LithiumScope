import json
from pathlib import Path

import pandas as pd

import lithiumscope.prediction.rebuild_pdf as rebuild_module


def test_rebuild_pdf_updates_existing_manifest(monkeypatch, tmp_path: Path):
    results = tmp_path / "results"
    run = results / "predictions" / "demonstration_test"
    (run / "model_1").mkdir(parents=True)
    (run / "model_2").mkdir(parents=True)
    (run / "cross_model" / "figures").mkdir(parents=True)

    monkeypatch.setattr(rebuild_module, "RESULTS_DIR", results)

    pd.DataFrame(
        [{
            "case_id": "x",
            "Li_icpms": 20.0,
            "Li_icpms_predicted": 18.0,
            "applicability_warning": "OK",
            "scientific_interpretation": "M1.",
        }]
    ).to_csv(run / "model_1" / "predictions.csv", index=False)
    pd.DataFrame(
        [{
            "case_id": "x",
            "Li_icpms": 20.0,
            "prospectivity_score": 0.3,
            "priority": "baja",
            "applicability_warning": "OK",
            "scientific_interpretation": "M2.",
        }]
    ).to_csv(run / "model_2" / "predictions.csv", index=False)
    pd.DataFrame(
        [{
            "case_id": "x",
            "Li_icpms": 20.0,
            "Li_icpms_predicted": 18.0,
            "prospectivity_score": 0.3,
            "priority": "baja",
        }]
    ).to_csv(run / "cross_model" / "joined_cases.csv", index=False)
    pd.DataFrame(
        [{
            "case_id": "x",
            "Li_icpms": 20.0,
            "Li_icpms_predicted": 18.0,
            "prospectivity_score": 0.3,
            "priority": "baja",
            "concordance": "concordante_baja",
            "scientific_interpretation": "Integrada.",
        }]
    ).to_csv(run / "cross_model" / "concordance.csv", index=False)
    (run / "interpretation.txt").write_text("Resumen.", encoding="utf-8")
    (run / "prediction_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "demonstration_test",
                "mode": "demonstration",
                "models": {
                    "model_1": {
                        "algorithm": "catboost",
                        "run_id": "training_a",
                        "model_sha256": "a" * 64,
                    },
                    "model_2": {
                        "algorithm": "random_forest",
                        "run_id": "training_b",
                        "model_sha256": "b" * 64,
                    },
                },
                "metrics": {
                    "model_1_external": {"rmse": 2.0},
                    "model_2_external": {"roc_auc": 0.6},
                },
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )

    report = rebuild_module.rebuild_pdf("demonstration_test")

    assert report.is_file()
    assert report.read_bytes().startswith(b"%PDF")
    manifest = json.loads(
        (run / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"]["report_pdf"].endswith("report.pdf")
