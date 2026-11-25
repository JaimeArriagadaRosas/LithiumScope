"""Production M2 feature preparation, with verified and content-addressed cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.model_2.data.feature_cache import source_identity, read_cache, write_cache
from lithiumscope.model_2.preparation.training_matrix import build_training_set


def _project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model_2_training_frame(path: Path | None = None) -> pd.DataFrame:
    cfg = load_config("model_2")
    training = cfg["training"]
    imagery = cfg["imagery"]
    manifest_path = path or _project_path(str(training["manifest_path"]))
    manifest, identity = source_identity(manifest_path, cfg)
    original = _project_path(str(training["manifest_path"]))
    configured = (training["feature_cache_path"] if manifest_path.resolve() == original.resolve()
                  else training.get("expansion", {}).get("feature_cache_path",
                        "data/processed/model_2/vnext_expanded/spectral_features.csv"))
    base = _project_path(str(configured))
    cache = base.parent / "verified_cache" / f"{canonical_json_hash(identity)}.csv"
    cached = read_cache(cache, manifest, cfg, identity)
    if cached is not None:
        print(f"[OK] Características espectrales verificadas reutilizadas: {len(cached)} muestras")
        return cached
    print("[INFO] No hay caché íntegra compatible; extrayendo características.")
    frame = build_training_set(
        manifest, lithium_column=str(training["lithium_column"]),
        spatial_group_column=str(training.get("spatial_group_column", "spatial_group")),
        band_names=tuple(str(value) for value in imagery["bands"]),
        normalize_per_band=bool(imagery.get("normalize_per_band", False)),
    )
    # Detect source changes during computation before publishing the cache.
    _, final_identity = source_identity(manifest_path, cfg)
    if identity != final_identity:
        raise ValueError("Spectral sources changed during feature extraction")
    write_cache(cache, frame, manifest, cfg, identity)
    return frame


def run_training(device, *, manifest=None):
    """M2 owns source resolution independently of M1 training."""
    from lithiumscope.runtime.preboot import require_full_training_environment
    from lithiumscope.runtime.dataset_provisioning import require_model_2_expanded_manifest
    from lithiumscope.model_2.training.competition import run_model_2_competition
    require_full_training_environment("model_2")
    source = manifest if manifest is not None else require_model_2_expanded_manifest()
    cfg=load_config("model_2")
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
        reserve(reservation_root,plan_path=_project_path(prospective["plan_path"]),
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


def run_independent_evaluation(phase, root, **kwargs):
    """Separate registered holdout phases; no automatic activation."""
    from lithiumscope.evaluation.holdout.workflow import run_holdout_phase
    return run_holdout_phase("model_2",phase,root,**kwargs)
