"""Separated evaluate_pending_candidates responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.prediction.candidates.evaluator import CandidateEvaluationResult

def evaluate_pending_candidates(
    *,
    promote_if_better: bool = True,
) -> CandidateEvaluationResult:
    from lithiumscope.prediction.candidates.evaluator import CandidateEvaluationResult, RESULTS_DIR, _evaluation_targets, _record_decision, _timestamp_with_offset, audit_demo_overlap, demonstration_imagery_cache, ensure_case_ids, joblib, json, load_config, load_demonstration_cases, model_1_external_metrics, model_1_gate, model_2_external_metrics, model_2_gate, predict_model_1_frame, prepare_case_images, score_model_2_cases
    evaluation_id = (
        "candidate_evaluation_"
        + _timestamp_with_offset()
    )
    root = (
        RESULTS_DIR
        / "candidate_evaluations"
        / evaluation_id
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = _evaluation_targets()
    decisions: dict[str, dict] = {}
    for model_group in (
        "model_1",
        "model_2",
    ):
        if model_group not in targets:
            decisions[model_group] = {
                "status": "not_evaluated",
                "reason": (
                    "falta modelo activo "
                    "o candidato pendiente"
                ),
            }

    cases = None
    overlap = None
    if targets:
        cases = ensure_case_ids(
            load_demonstration_cases()
        )
        overlap = audit_demo_overlap(
            cases
        )

    if (
        cases is not None
        and "model_1" in targets
    ):
        active_path, candidate_path = (
            targets["model_1"]
        )
        active_bundle = joblib.load(
            active_path
        )
        candidate_bundle = joblib.load(
            candidate_path
        )
        active_predictions, _ = (
            predict_model_1_frame(
                cases,
                bundle=active_bundle,
            )
        )
        candidate_predictions, _ = (
            predict_model_1_frame(
                cases,
                bundle=candidate_bundle,
            )
        )
        active_metrics = (
            model_1_external_metrics(
                active_predictions
            )
        )
        candidate_metrics = (
            model_1_external_metrics(
                candidate_predictions
            )
        )
        passed, reasons = model_1_gate(
            active_metrics,
            candidate_metrics,
        )
        group_dir = root / "model_1"
        group_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        active_predictions.to_csv(
            group_dir
            / "active_predictions.csv",
            index=False,
        )
        candidate_predictions.to_csv(
            group_dir
            / "candidate_predictions.csv",
            index=False,
        )
        decisions["model_1"] = (
            _record_decision(
                model_group="model_1",
                active_path=active_path,
                candidate_path=candidate_path,
                active_metrics=active_metrics,
                candidate_metrics=candidate_metrics,
                passed=passed,
                reasons=reasons,
                evaluation_id=evaluation_id,
                promote_if_better=(
                    promote_if_better
                ),
            )
        )

    if (
        cases is not None
        and "model_2" in targets
    ):
        active_path, candidate_path = (
            targets["model_2"]
        )
        active_bundle = joblib.load(
            active_path
        )
        candidate_bundle = joblib.load(
            candidate_path
        )
        demo_cfg = load_config(
            "prediction"
        )["demonstration"]
        imagery = prepare_case_images(
            cases,
            demonstration_imagery_cache(),
            datetime_override=str(
                demo_cfg[
                    "sentinel_datetime"
                ]
            ),
        )
        active_predictions = (
            score_model_2_cases(
                imagery,
                bundle=active_bundle,
            )
        )
        candidate_predictions = (
            score_model_2_cases(
                imagery,
                bundle=candidate_bundle,
            )
        )
        active_reference = float(
            active_bundle[
                "threshold_ppm"
            ]
        )
        candidate_reference = float(
            candidate_bundle[
                "threshold_ppm"
            ]
        )
        active_metrics = (
            model_2_external_metrics(
                active_predictions,
                threshold_ppm=(
                    active_reference
                ),
                probability_threshold=float(
                    active_bundle.get(
                        "operating_threshold",
                        0.5,
                    )
                ),
            )
        )
        candidate_metrics = (
            model_2_external_metrics(
                candidate_predictions,
                threshold_ppm=(
                    candidate_reference
                ),
                probability_threshold=float(
                    candidate_bundle.get(
                        "operating_threshold",
                        0.5,
                    )
                ),
            )
        )
        passed, reasons = model_2_gate(
            active_metrics,
            candidate_metrics,
            same_reference_threshold=(
                abs(
                    active_reference
                    - candidate_reference
                )
                <= 1e-12
            ),
        )
        group_dir = root / "model_2"
        group_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        active_predictions.to_csv(
            group_dir
            / "active_predictions.csv",
            index=False,
        )
        candidate_predictions.to_csv(
            group_dir
            / "candidate_predictions.csv",
            index=False,
        )
        decisions["model_2"] = (
            _record_decision(
                model_group="model_2",
                active_path=active_path,
                candidate_path=candidate_path,
                active_metrics=active_metrics,
                candidate_metrics=candidate_metrics,
                passed=passed,
                reasons=reasons,
                evaluation_id=evaluation_id,
                promote_if_better=(
                    promote_if_better
                ),
            )
        )

    manifest = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "role": (
            "known_external_acceptance_set; "
            "not a blind validation after model changes"
        ),
        "overlap_audit": overlap,
        "decisions": decisions,
    }
    manifest_path = (
        root
        / "candidate_evaluation.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return CandidateEvaluationResult(
        evaluation_id=evaluation_id,
        root=root,
        manifest_path=manifest_path,
        decisions=decisions,
    )
