"""Reserve and score independent evidence around the production competition."""
from pathlib import Path

from lithiumscope.core.paths import resolve_project_path


def train_with_prospective_evidence(source, device, cfg):
    from lithiumscope.model_2.training.competition import run_model_2_competition
    prospective=cfg.get("prospective_evaluation",{})
    from lithiumscope.core.paths import RESULTS_DIR, MODELS_DIR
    from lithiumscope.results.atomic_exports import write_json
    from lithiumscope.core.logging.context import unique_id
    reservation_root=None
    previous_candidates=set()
    if prospective.get("enabled",False):
        if prospective.get("trusted_local_bundles") is not True:
            raise ValueError("Prospective evaluation requires explicit trust in local bundles")
        from lithiumscope.evaluation.holdout.reservation import reserve
        reservation_root=RESULTS_DIR/"prospective_m2"/unique_id("reservation")
        reserve(reservation_root,plan_path=resolve_project_path(prospective["plan_path"]),
                manifest_path=source,fixed_endpoint=cfg["model"].get("fixed_threshold_ppm"))
        previous_candidates={p.resolve() for p in (MODELS_DIR/"model_2"/"trained").glob("*/model.joblib")}
    outcome=run_model_2_competition(source, device)
    status={"state":"blocked_no_reserved_holdout","scientific_release_evidence_pass":False,"automatic_model_activation":False}
    if reservation_root is not None:
        status["reservation"]=str(reservation_root)
        try:
            import json
            from lithiumscope.evaluation.holdout.reservation import bind_and_score
            summary=json.loads((outcome.run_dir/"run.json").read_text(encoding="utf-8"))["summary"]
            candidate=Path(summary["model_path"]).resolve(strict=True)
            if not outcome.winner or candidate in previous_candidates:
                raise ValueError("A newly selected candidate is required for this pre-training reservation")
            bind_and_score(reservation_root,candidate,trusted_local_bundles=True)
            status.update(state="awaiting_independent_label_reveal",holdout=str(reservation_root/"holdout"))
        except Exception as exc:
            from lithiumscope.core.logging.session import get_logger
            get_logger("model_2.pipeline").exception("Prospective evaluation blocked")
            status.update(state="blocked_prospective_evaluation",error_type=type(exc).__name__)
    write_json(outcome.run_dir/"prospective_status.json",status)
    return outcome
