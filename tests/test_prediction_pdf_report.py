from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from reportlab.platypus import PageBreak

from lithiumscope.prediction.interpretation import (
    integrated_case_text,
    model_1_case_text,
    model_2_case_text,
)
from lithiumscope.prediction.report import write_pdf_report
from lithiumscope.prediction.reporting.charts import (
    _satellite_preview_grid_shape,
    save_satellite_input_preview,
)
from lithiumscope.prediction.reporting.formatting import (
    nonnegative_interval,
    operating_classification,
)
from lithiumscope.prediction.reporting.sections.overview import (
    _out_of_domain_count,
)
from lithiumscope.prediction.reporting.sections.spatial import (
    build_spatial,
)
from lithiumscope.prediction.reporting.styles import build_styles


def test_pdf_report_is_generated_with_case_sections(tmp_path: Path):
    m1 = pd.DataFrame(
        [
            {
                "case_id": "case_a",
                "Li_icpms": 25.0,
                "Li_icpms_predicted": 8.0,
                "Li_icpms_interval_low_q90": -6.94,
                "Li_icpms_interval_high_q90": 22.03,
                "out_of_training_range_fraction": 0.1,
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
                "prospectivity_score": 0.175,
                "priority": "baja",
                "operating_positive": True,
                "operating_threshold": 0.088,
                "out_of_training_range_fraction": 0.3,
                "applicability_warning": "OUT_OF_DOMAIN",
                "sentinel_scene_id": "scene_a",
                "sentinel_cloud_cover": 3.5,
                "sentinel_datetime": "2025-01-01T10:00:00Z",
                "scientific_interpretation": "Interpretacion M2.",
            }
        ]
    )
    paired = m1[["case_id", "Li_icpms", "Li_icpms_predicted"]].merge(
        m2[["case_id", "prospectivity_score", "priority"]],
        on="case_id",
    )
    concordance = paired.copy()
    concordance["concordance"] = "divergente_m2_alto"
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
            [
                {
                    "relationship": "Li real <-> score M2",
                    "n": 1,
                    "pearson": None,
                    "spearman": None,
                }
            ]
        ),
        concordance=concordance,
        training_vs_external=pd.DataFrame(),
        overlap_audit={
            "status": "sin coincidencias",
            "sample_id_matches": 0,
            "coordinate_matches": 0,
        },
        input_info={"demonstration": {"source_name": "external"}},
        model_1_diagnostics={
            "out_of_domain_rows": 0,
            "missing_expected_columns": [],
        },
        model_2_diagnostics={
            "failed_cases": 0,
            "missing_feature_counts": {},
        },
        maps=[],
        figures=[],
        lithium_threshold_ppm=20.0,
        model_2_classification_threshold=0.088,
        runtime={"git_commit": "abc123"},
    )

    payload = destination.read_bytes()
    assert payload.startswith(b"%PDF")
    assert len(payload) > 1000


def test_satellite_input_preview_uses_four_column_grid_and_real_multiband_patches(
    tmp_path: Path,
):
    rows = []
    for index in range(10):
        patch = np.zeros((6, 8, 8), dtype=np.float32)
        gradient = np.linspace(
            0.0,
            1.0,
            64,
            dtype=np.float32,
        ).reshape(8, 8)
        patch[0] = gradient + index
        patch[1] = gradient.T + index
        patch[2] = np.flipud(gradient) + index
        patch_path = tmp_path / f"case_{index}.npy"
        np.save(patch_path, patch)
        rows.append(
            {
                "case_id": f"case_{index}",
                "image_path": str(patch_path),
                "sentinel_status": "ready",
            }
        )

    assert _satellite_preview_grid_shape(10) == (3, 4)

    destination = tmp_path / "sentinel_inputs.png"
    result = save_satellite_input_preview(
        pd.DataFrame(rows),
        destination,
    )

    assert result == destination
    assert destination.is_file()
    assert destination.stat().st_size > 1000


def test_lithium_interval_is_clamped_only_for_presentation():
    assert (
        nonnegative_interval(-6.94, 22.03, 2)
        == "0.00 a 22.03 ppm"
    )

    text = model_1_case_text(
        pd.Series(
            {
                "Li_icpms_predicted": 8.0,
                "Li_icpms_interval_low_q90": -6.94,
                "Li_icpms_interval_high_q90": 22.03,
                "out_of_training_range_fraction": 0.0,
            }
        )
    )
    assert "0.00 a 22.03 ppm" in text


def test_model_2_descriptive_category_is_distinct_from_operating_classification():
    row = pd.Series(
        {
            "prospectivity_score": 0.175,
            "priority": "baja",
            "operating_threshold": 0.088,
            "out_of_training_range_fraction": 0.0,
        }
    )
    text = model_2_case_text(row)

    assert "categoría descriptiva del score BAJA" in text
    assert "clasificación es POSITIVA" in text
    assert "(0.088)" in text
    assert operating_classification(0.175, 0.088) == "POSITIVA"

    integrated = integrated_case_text(
        pd.Series(
            {
                "concordance": "divergente_m2_alto",
                "model_1_out_of_training_range_fraction": 0.0,
                "model_2_out_of_training_range_fraction": 0.0,
            }
        )
    )
    assert "supera el umbral operativo del Modelo 2" in integrated
    assert "señal espacial/espectral elevada" not in integrated


def test_spatial_section_breaks_immediately_before_coverage():
    context = SimpleNamespace(
        title="LithiumScope - Demo",
        run_id="demo",
        mode="demonstration",
        interpretation=(
            "INTERPRETACIÓN CIENTÍFICA INTEGRADA\n"
            "\n"
            "Pregunta del Modelo 1: uno\n"
            "Pregunta del Modelo 2: dos\n"
            "\n"
            "Cobertura de la ejecución:\n"
            "- Casos evaluados por Modelo 1: 10"
        ),
        figures=(),
        maps=(),
    )
    story = build_spatial(context, build_styles())

    coverage_index = next(
        index
        for index, item in enumerate(story)
        if getattr(item, "getPlainText", lambda: "")()
        == "Cobertura de la ejecución:"
    )
    assert isinstance(story[coverage_index - 1], PageBreak)


def test_model_2_ood_count_uses_applicability_warning():
    frame = pd.DataFrame(
        [
            {"applicability_warning": "OUT_OF_DOMAIN"},
            {"applicability_warning": "OK"},
            {"applicability_warning": "OUT_OF_DOMAIN"},
        ]
    )
    assert _out_of_domain_count(frame) == 2
