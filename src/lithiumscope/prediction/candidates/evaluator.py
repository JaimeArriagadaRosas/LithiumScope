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
        "variant_id": metadata.get(
            "variant_id"
        ),
        "target_transform": metadata.get(
            "target_transform",
            "identity",
        ),
        "feature_extractor_version": metadata.get(
            "feature_extractor_version"
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


from lithiumscope.prediction.candidates.evaluator_evaluate_pending_candidates import evaluate_pending_candidates
