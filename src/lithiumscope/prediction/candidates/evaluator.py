from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import joblib

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.model_1.prediction.predictor import (
    predict_model_1_frame,
)
from lithiumscope.persistence.active_models import (
    active_model_path,
    set_active_model,
)
from lithiumscope.persistence.candidates import (
    latest_pending_candidate,
    update_candidate_status,
)
from lithiumscope.prediction.analysis import (
    ensure_case_ids,
    model_1_external_metrics,
    model_2_external_metrics,
)
from lithiumscope.prediction.candidates.gates import (
    model_1_gate,
    model_2_gate,
)
from lithiumscope.prediction.candidates.scoring import (
    score_model_2_cases,
)
from lithiumscope.prediction.demo import (
    audit_demo_overlap,
    demonstration_imagery_cache,
    load_demonstration_cases,
)
from lithiumscope.prediction.imagery import (
    prepare_case_images,
)


@dataclass(frozen=True)
class CandidateEvaluationResult:
    evaluation_id: str
    root: Path
    manifest_path: Path
    decisions: dict[str, dict]


def _timestamp_with_offset() -> str:
    now = datetime.now().astimezone()
    offset = now.strftime("%z")
    sign = (
        "p"
        if offset.startswith("+")
        else "m"
    )
    digits = (
        offset[1:]
        if offset
        else "0000"
    )
    return (
        now.strftime("%Y%m%d_%H%M%S")
        + f"_{sign}{digits}"
    )


def _metadata(
    model_path: Path,
) -> dict:
    path = (
        model_path.parent
        / "metadata.json"
    )
    if not path.is_file():
        return {}
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}
    return (
        payload
        if isinstance(payload, dict)
        else {}
    )


def _identity(
    model_path: Path,
) -> dict:
    metadata = _metadata(
        model_path
    )
    return {
        "run_id": metadata.get(
            "run_id",
            model_path.parent.name,
        ),
        "algorithm": metadata.get(
            "algorithm",
            "unknown",
        ),
        "model_path": str(model_path),
        "model_sha256": file_sha256(
            model_path
        ),
    }


def _evaluation_targets() -> dict[str, tuple[Path, Path]]:
    targets: dict[str, tuple[Path, Path]] = {}
    for model_group in (
        "model_1",
        "model_2",
    ):
        active = active_model_path(
            model_group
        )
        candidate = latest_pending_candidate(
            model_group
        )
        if (
            active is not None
            and candidate is not None
        ):
            targets[model_group] = (
                active,
                candidate,
            )
    return targets


def _record_decision(
    *,
    model_group: str,
    active_path: Path,
    candidate_path: Path,
    active_metrics: dict,
    candidate_metrics: dict,
    passed: bool,
    reasons: list[str],
    evaluation_id: str,
    promote_if_better: bool,
) -> dict:
    promoted = bool(
        passed
        and promote_if_better
    )
    if promoted:
        set_active_model(
            model_group,
            candidate_path,
            source={
                "type": (
                    "candidate_evaluation"
                ),
                "evaluation_id": (
                    evaluation_id
                ),
            },
        )

    status = (
        "promoted"
        if promoted
        else "accepted_not_promoted"
        if passed
        else "rejected"
    )
    decision = {
        "status": status,
        "passed_gate": passed,
        "promoted": promoted,
        "reasons": reasons,
        "active": _identity(
            active_path
        ),
        "candidate": _identity(
            candidate_path
        ),
        "active_external_metrics": (
            active_metrics
        ),
        "candidate_external_metrics": (
            candidate_metrics
        ),
    }
    update_candidate_status(
        candidate_path,
        status=(
            "promoted"
            if promoted
            else "accepted"
            if passed
            else "rejected"
        ),
        evaluation_id=evaluation_id,
        details=decision,
    )
    return decision


def evaluate_pending_candidates(
    *,
    promote_if_better: bool = True,
) -> CandidateEvaluationResult:
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
