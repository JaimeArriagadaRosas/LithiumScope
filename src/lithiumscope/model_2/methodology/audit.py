"""Opt-in CLI to audit a frozen Model 2 experimental protocol.

Example:
  python -m lithiumscope.model_2.methodology.audit \
     --manifest data/processed/model_2/vnext_expanded/training_manifest.csv \
     --features data/processed/model_2/vnext_expanded/spectral_features.csv \
     --folds results/model_2/runs/<run>/manifests/scientific_folds.json \
     --output results/scientific_simulations/<new-run>/phase_b_audit

It never trains a model or changes historical folds. Strict validation is opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.contracts import (
    sample_contract, policy_counts, site_summary,
)
from lithiumscope.model_2.methodology.spatial import (
    assess_folds, buffered_train_preview,
)
from lithiumscope.model_2.methodology.domain import (
    training_domain_profile, score_domain,
)


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_audit(
    manifest_path: Path, features_path: Path, fold_path: Path,
    output_dir: Path, *, buffer_m: float = 5000, strict: bool = False
) -> dict:
    """Inspect pre-existing data and folds without altering any of them."""
    for item in (manifest_path, features_path, fold_path):
        if not item.is_file():
            raise FileNotFoundError(item)
    contract = sample_contract(pd.read_csv(manifest_path))
    folds = json.loads(fold_path.read_text(encoding="utf-8"))
    if not isinstance(folds, list):
        raise ValueError("Fold manifest must be a list")
    data = pd.read_csv(features_path)
    if len(data) != len(contract):
        raise ValueError("Feature cache row count differs from manifest")
    feature_names = [c for c in data.columns if c not in ("Li_icpms", "spatial_group")]
    if len(feature_names) == 0:
        raise ValueError("Feature cache contains no numeric predictors")
    # Fail closed on silent sample reordering: full row-level coordinates are
    # unavailable in the cache, so check the Li and spatial-group vectors.
    if "Li_icpms" not in data or "spatial_group" not in data:
        raise ValueError("Feature cache is missing identity cross-check columns")
    if not (
        pd.to_numeric(data.Li_icpms).round(9).reset_index(drop=True).equals(
            pd.to_numeric(contract.Li_icpms).round(9).reset_index(drop=True)
        ) and data.spatial_group.astype(str).reset_index(drop=True).equals(
            contract.spatial_group.astype(str).reset_index(drop=True)
        )
    ):
        raise ValueError("Feature cache rows no longer align with manifest")
    features = data[feature_names]
    counts = policy_counts(contract)
    fold_summary, rows = assess_folds(contract, folds, warning_distance_m=buffer_m)
    buffered = buffered_train_preview(contract, folds, buffer_m=buffer_m)
    # A scene-only exclusion is a diagnostic alternative, NOT the default policy:
    scene_only = buffered_train_preview(
        contract, folds, buffer_m=buffer_m, exclude_shared_scenes=True
    )
    if any(v < 1 for v in buffered.n_train_remaining):
        raise ValueError("No train rows after buffer exclusion preview")
    fold_ood = []
    profiles = []
    for index, fold in enumerate(folds, 1):
        train = [int(i) for i in fold["train_indices"]]
        test = [int(i) for i in fold["test_indices"]]
        profile = training_domain_profile(features.iloc[train])
        score = score_domain(features.iloc[test], profile)
        profiles.append({"fold":index, "profile":profile})
        for sample_index, (_, row) in zip(test, score.iterrows()):
            fold_ood.append({
                "fold": index,
                "sample_id": str(contract.sample_id.iloc[sample_index]),
                "test_index": sample_index,
                "ood_fraction": float(row.ood_fraction),
                "ood_alert": bool(row.ood_alert),
                "features_outside": int(row.features_outside),
                "has_nonfinite_input": bool(row.has_nonfinite_input),
            })
    ood_frame = pd.DataFrame(fold_ood)
    if len(ood_frame) != len(contract) or ood_frame.test_index.nunique() != len(contract):
        raise ValueError("Fold domain scoring did not cover each sample exactly once")
    # No external/demo cases used here.
    strict_issues = []
    if counts["conflicting_sites"]:
        strict_issues.append(f"{counts['conflicting_sites']} sites have conflicting Li classes")
    if counts["excluded_samples"]:
        strict_issues.append(f"{counts['excluded_samples']} ineligible samples")
    if counts["historical_without_explicit_field_verification"]:
        strict_issues.append("Historical labels lack explicit field-verification metadata")
    if counts["missing_scene_id"]:
        strict_issues.append("Some scene IDs are missing; scene independence cannot be audited")
    if counts["missing_publication_id"]:
        strict_issues.append("Some publication IDs are missing; bibliographic independence cannot be audited")
    if counts["eligibility"].get("PROVISIONAL_SURFACE", 0):
        strict_issues.append("Provisional GEOROC labels are unverified in the field")
    if int(fold_summary.n_test_too_close.sum()):
        strict_issues.append("Test samples violate the requested geodesic buffer")
    if int(fold_summary.n_test_scene_shared.sum()):
        strict_issues.append("Training and test observations share Sentinel-2 scenes")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract.to_csv(output_dir / "sample_provenance_contract.csv", index=False)
    site_summary(contract).to_csv(output_dir / "site_conflicts.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_dependency_summary.csv", index=False)
    rows.to_csv(output_dir / "fold_dependency_samples.csv", index=False)
    ood_frame.to_csv(output_dir / "fold_train_only_domain_scores.csv", index=False)
    # Preserve buffer exclusions for later development; no execution changes.
    for name, frame in (
        ("buffer_training_preview.csv", buffered),
        ("buffer_and_scene_preview.csv", scene_only),
    ):
        frame.drop(columns=["train_indices_remaining", "test_indices_unchanged"]).to_csv(
            output_dir / name, index=False
        )
    (output_dir / "buffered_index_preview.json").write_text(
        json.dumps([
            {"fold":int(row.fold),
             "train_indices_remaining":list(row.train_indices_remaining),
             "test_indices_unchanged":list(row.test_indices_unchanged)}
            for row in buffered.itertuples(index=False)
        ], indent=2), encoding="utf-8"
    )
    (output_dir / "fold_train_only_domain_profiles.json").write_text(
        json.dumps(profiles, indent=2), encoding="utf-8"
    )
    summary = {
        "schema_version":1,
        "purpose":"phase_b_methodological_preflight",
        "model_fitted":False,
        "menu_modified":False,
        "original_folds_modified":False,
        "input_sha256":{
            "manifest":_sha(manifest_path),
            "features":_sha(features_path),
            "folds":_sha(fold_path),
        },
        "labels":counts,
        "folds":len(folds),
        "buffer_warning_m":buffer_m,
        "test_rows_near_train":int(fold_summary.n_test_too_close.sum()),
        "test_rows_sharing_scene":int(fold_summary.n_test_scene_shared.sum()),
        "test_rows_sharing_publication":int(fold_summary.n_test_publication_shared.sum()),
        "buffered_train_min":int(buffered.n_train_remaining.min()),
        "buffered_train_removed_total":int(buffered.n_removed_buffer.sum()),
        "buffer_and_scene_train_min":int(scene_only.n_train_remaining.min()),
        "fold_ood_alerts":int(ood_frame.ood_alert.sum()),
        "strict_policy_pass":not strict_issues,
        "strict_policy_issues":strict_issues,
        "strict_requested":bool(strict),
        "note":"Proximity and same-scene events indicate dependence, not proven target leakage.",
    }
    (output_dir / "phase_b_preflight.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--buffer-m", type=float, default=5000)
    p.add_argument("--strict", action="store_true", help="Nonzero exit if methodological issues remain")
    args = p.parse_args(argv)
    result = run_audit(
        _path(args.manifest),_path(args.features),_path(args.folds),
        _path(args.output),buffer_m=args.buffer_m,strict=args.strict,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and not result["strict_policy_pass"]:
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
