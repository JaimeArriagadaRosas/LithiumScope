"""Orchestrate integrated inference and scientific analysis."""
from __future__ import annotations
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.logging.session import get_logger, run_log
from lithiumscope.model_1.prediction.predictor import predict_model_1_frame
from lithiumscope.prediction.inputs.case_identity import ensure_case_ids
from lithiumscope.prediction.analysis.metrics import model_1_external_metrics, model_2_external_metrics
from lithiumscope.prediction.analysis.comparison import (
    concordance_table, correlation_table, pair_model_outputs, training_vs_external_table,
)
from lithiumscope.prediction.analysis.interpretation import integrated_case_text, model_1_case_text
from lithiumscope.prediction.analysis.interpretation_summary import build_overall_interpretation
from lithiumscope.prediction.execution.session import PredictionSession, load_model_with_identity
from lithiumscope.prediction.execution.contracts import (
    IntegratedPredictionResult, PredictionFrames, PredictionEvidence, PredictionThresholds,
)
from lithiumscope.prediction.execution.model_2_scoring import _score_model_2_cases
from lithiumscope.prediction.execution.outputs import _write_outputs

logger = get_logger("prediction.integrated")


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

        frames = PredictionFrames(model_1_predictions, model_2_predictions, model_2_cases,
                                  paired, correlations, concordance, training_vs_external)
        evidence = PredictionEvidence(model_1_metrics, model_2_metrics, model_1_identity,
                                      model_2_identity, model_1_diagnostics, model_2_diagnostics,
                                      input_info, overlap_audit, interpretation)
        thresholds = PredictionThresholds(lithium_threshold, probability_threshold,
                                          model_2_predicted_positive_count)
        report_path, workbook_path, manifest_path = _write_outputs(
            session, frames, evidence, thresholds, log_path,
        )
        logger.info(
            "Integrated prediction finished run=%s report=%s workbook=%s",
            session.run_id,
            report_path,
            workbook_path,
        )

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
