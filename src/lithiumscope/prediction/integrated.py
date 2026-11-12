from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logger import get_logger, run_log
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.model_1.prediction.input_parser import load_prediction_input
from lithiumscope.model_1.prediction.predictor import predict_model_1_frame
from lithiumscope.model_2.prediction.image_parser import image_to_feature_frame
from lithiumscope.model_2.schema import out_of_range_fraction
from lithiumscope.prediction.analysis import (
    concordance_table,
    correlation_table,
    ensure_case_ids,
    model_1_external_metrics,
    model_2_external_metrics,
    pair_model_outputs,
    training_vs_external_table,
)
from lithiumscope.prediction.demo import (
    audit_demo_overlap,
    demonstration_imagery_cache,
    load_demonstration_cases,
)
from lithiumscope.prediction.imagery import prepare_case_images
from lithiumscope.prediction.interpretation import (
    build_overall_interpretation,
    integrated_case_text,
    model_1_case_text,
    model_2_case_text,
)
from lithiumscope.prediction.report import (
    save_integrated_figures,
    save_satellite_input_preview,
    save_spatial_maps,
    write_json,
    write_pdf_report,
    write_workbook,
)
from lithiumscope.prediction.session import (
    PredictionSession,
    load_model_with_identity,
)

logger = get_logger("prediction.integrated")


@dataclass(frozen=True)
class IntegratedPredictionResult:
    session: PredictionSession
    model_1_predictions: pd.DataFrame
    model_2_predictions: pd.DataFrame
    paired: pd.DataFrame
    correlations: pd.DataFrame
    concordance: pd.DataFrame
    model_1_metrics: dict
    model_2_metrics: dict
    interpretation: str
    report_path: Path
    workbook_path: Path
    manifest_path: Path


_IMAGE_EXTENSIONS = {".tif", ".tiff", ".npy"}
_TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _missing_expected_columns(frame: pd.DataFrame, schema) -> list[str]:
    expected = list(schema.numeric) + list(schema.categorical)
    return [column for column in expected if column not in frame.columns]


from lithiumscope.prediction.integrated_score_model_2_cases import _score_model_2_cases


from lithiumscope.prediction.integrated_model_2_cases_from_input import _model_2_cases_from_input


from lithiumscope.prediction.integrated_write_outputs import _write_outputs


def _run_integrated(
    *,
    session: PredictionSession,
    model_1_input: pd.DataFrame,
    model_2_cases: pd.DataFrame,
    input_info: dict,
    overlap_audit: dict | None = None,
) -> IntegratedPredictionResult:
    prediction_cfg = load_config("prediction")["interpretation"]
    fallback_probability_threshold = float(
        prediction_cfg["model_2_classification_threshold"]
    )
    local_log_path = session.root / "prediction.log"

    with run_log("prediction", session.run_id) as log_path:
        logger.info(
            "Integrated prediction started run=%s mode=%s",
            session.run_id,
            session.mode,
        )

        # Demonstration only: use the explicitly pinned new candidates.
        # All other prediction routes continue to load the published active models.
        demonstration_candidates = (
            load_config("prediction")["demonstration"]["candidate_models"]
            if session.mode == "demonstration"
            else {}
        )
        model_1_bundle, _, model_1_identity = load_model_with_identity(
            "model_1",
            candidate_run_id=demonstration_candidates.get("model_1"),
        )
        model_2_bundle, _, model_2_identity = load_model_with_identity(
            "model_2",
            candidate_run_id=demonstration_candidates.get("model_2"),
        )
        logger.info(
            "Models loaded model_1=%s model_2=%s",
            model_1_identity["run_id"],
            model_2_identity["run_id"],
        )
        lithium_threshold = float(model_2_bundle["threshold_ppm"])
        probability_threshold = float(
            model_2_bundle.get(
                "operating_threshold",
                fallback_probability_threshold,
            )
        )

        raw_m1 = ensure_case_ids(model_1_input)
        logger.info(
            "Model 1 prediction stage started cases=%d",
            len(raw_m1),
        )
        model_1_predictions, prediction_diagnostics = (
            predict_model_1_frame(
                raw_m1,
                bundle=model_1_bundle,
            )
        )
        model_1_predictions["scientific_interpretation"] = (
            model_1_predictions.apply(
                lambda row: model_1_case_text(
                    row,
                    lithium_threshold_ppm=lithium_threshold,
                ),
                axis=1,
            )
        )
        model_1_diagnostics = {
            **prediction_diagnostics,
            "input_rows": len(raw_m1),
        }
        logger.info(
            "Model 1 prediction stage completed cases=%d ood_rows=%s",
            len(model_1_predictions),
            model_1_diagnostics.get("out_of_domain_rows"),
        )

        logger.info(
            "Model 2 prediction stage started requested_cases=%d",
            len(model_2_cases),
        )
        model_2_predictions, model_2_diagnostics = _score_model_2_cases(
            model_2_cases,
            model_2_bundle,
        )
        logger.info(
            "Model 2 prediction stage completed ready=%d failed=%d",
            len(model_2_predictions),
            model_2_diagnostics["failed_cases"],
        )

        paired = pair_model_outputs(
            model_1_predictions,
            model_2_predictions,
        )
        correlations = correlation_table(paired)
        concordance = concordance_table(
            paired,
            lithium_threshold_ppm=lithium_threshold,
            model_2_probability_threshold=probability_threshold,
        )
        if not concordance.empty:
            concordance["scientific_interpretation"] = concordance.apply(
                integrated_case_text,
                axis=1,
            )

        model_1_metrics = model_1_external_metrics(
            model_1_predictions
        )
        model_2_metrics = model_2_external_metrics(
            model_2_predictions,
            threshold_ppm=lithium_threshold,
            probability_threshold=probability_threshold,
        )
        training_vs_external = training_vs_external_table(
            model_1_identity,
            model_2_identity,
            model_1_metrics,
            model_2_metrics,
        )
        model_2_scores = pd.to_numeric(
            model_2_predictions.get(
                "prospectivity_score",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()
        model_2_predicted_positive_count = int(
            (model_2_scores >= probability_threshold).sum()
        )
        interpretation = build_overall_interpretation(
            model_1_metrics=model_1_metrics,
            model_2_metrics=model_2_metrics,
            correlations=correlations,
            concordance=concordance,
            overlap_audit=overlap_audit,
            model_1_case_count=len(model_1_predictions),
            model_2_case_count=len(model_2_predictions),
            paired_case_count=len(paired),
            model_2_predicted_positive_count=model_2_predicted_positive_count,
        )
        logger.info(
            "Cross-model analysis completed paired=%d correlations=%d",
            len(paired),
            len(correlations),
        )

        report_path, workbook_path, manifest_path = _write_outputs(
            session=session,
            model_1_predictions=model_1_predictions,
            model_2_predictions=model_2_predictions,
            model_2_cases=model_2_cases,
            paired=paired,
            correlations=correlations,
            concordance=concordance,
            training_vs_external=training_vs_external,
            model_1_metrics=model_1_metrics,
            model_2_metrics=model_2_metrics,
            model_1_identity=model_1_identity,
            model_2_identity=model_2_identity,
            model_1_diagnostics=model_1_diagnostics,
            model_2_diagnostics=model_2_diagnostics,
            input_info=input_info,
            overlap_audit=overlap_audit,
            interpretation=interpretation,
            run_log_path=local_log_path,
            lithium_threshold=lithium_threshold,
            probability_threshold=probability_threshold,
            model_2_predicted_positive_count=model_2_predicted_positive_count,
        )
        logger.info(
            "Integrated prediction finished run=%s report=%s workbook=%s",
            session.run_id,
            report_path,
            workbook_path,
        )

    shutil.copy2(log_path, local_log_path)

    return IntegratedPredictionResult(
        session=session,
        model_1_predictions=model_1_predictions,
        model_2_predictions=model_2_predictions,
        paired=paired,
        correlations=correlations,
        concordance=concordance,
        model_1_metrics=model_1_metrics,
        model_2_metrics=model_2_metrics,
        interpretation=interpretation,
        report_path=report_path,
        workbook_path=workbook_path,
        manifest_path=manifest_path,
    )

from lithiumscope.prediction.integrated_run_complete_prediction import run_complete_prediction

from lithiumscope.prediction.integrated_run_automatic_demonstration import run_automatic_demonstration
