"""M1 development-only experiments on existing GEOROC and Mamani09 data.

Builds strict FeOT-reconstructed candidate inputs without changing any canonical
dataset, running group-heldout comparisons or promoting a bundle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.datasets.georoc_ingestion_schema import infer_georoc_column_map
from lithiumscope.datasets.georoc_ingestion_transform import harmonize_chunk
from lithiumscope.datasets.harmonization import (
    canonicalize_coordinates, merge_harmonized_sources,
)
from lithiumscope.experiments.analysis.georoc_qc_protocols import protocol_masks
from lithiumscope.model_1.pipeline import prepare_prediction_data
from lithiumscope.model_1.steps.step_02_detection_limits import clean_detection_limits
from lithiumscope.model_1.steps.step_03_missing_values import normalize_missing_values
from lithiumscope.model_1.steps.step_09_validation import build_spatial_groups
from lithiumscope.model_1.training.target_transform import wrap_target_regressor


IRON_FEOT_TO_FE2O3T = 1.111348


def harmonize_feot_candidate(raw: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Convert total Fe as FeO to total Fe as Fe2O3 in an experimental copy.

    FEOT is TOTAL iron; never add it to any FE2O3 column. If FEOT is absent,
    the original Fe2O3 value can be used, but provenance is explicit.
    """
    copy = raw.copy(deep=True)
    feot = pd.to_numeric(copy.get("FEOT(WT%)"), errors="coerce")
    ferric = pd.to_numeric(copy.get("FE2O3(WT%)"), errors="coerce")
    if not isinstance(feot, pd.Series) or not isinstance(ferric, pd.Series):
        raise ValueError("Required raw Fe source columns are not present.")
    candidate = feot.mul(IRON_FEOT_TO_FE2O3T).where(feot.notna(), ferric)
    copy["FE2O3T(WT%)"] = candidate
    mapping = infer_georoc_column_map(copy.columns)
    if mapping.get("FE2O3T(WT%)") != "Fe2O3":
        raise RuntimeError("The GEOROC schema did not select the total iron column.")
    harmonized, _ = harmonize_chunk(
        copy,
        column_map=mapping,
        source_name="GEOROC",
        source_file=source_file,
        allowed_material_types=("WHOLE ROCK",),
        minimum_predictors=8,
    )
    harmonized["Fe_total_provenance"] = np.where(
        feot.reindex(harmonized.index).notna(),
        "FEOT_CONVERTED_TO_FE2O3T",
        "ORIGINAL_FE2O3",
    )
    return harmonized


def quality_candidate_frames(
    source: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Prepare three variants in-memory using the existing preprocessing rules."""
    frame = canonicalize_coordinates(
        normalize_missing_values(clean_detection_limits(source))
    )
    frame["Li_icpms"] = pd.to_numeric(frame["Li_icpms"], errors="coerce")
    frame = frame.dropna(subset=["Li_icpms"]).copy()
    masks = protocol_masks(frame)
    return {
        name: frame.loc[masks[flag]].copy()
        for name, flag in (
            ("B", "B_reconstructed"),
            ("C", "C_experimental"),
        )
    }


def heldout_group_masks(
    base: pd.DataFrame,
    extended: pd.DataFrame,
    *,
    degrees: float = 0.5,
    splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return shared test rows from A and B/C train indices, no overlapping group."""
    a_groups = build_spatial_groups(base, degrees=degrees)
    b_groups = build_spatial_groups(extended, degrees=degrees)
    if a_groups is None or b_groups is None:
        raise ValueError("Valid spatial coordinates required for M1 experiments.")
    pairs = []
    for _, test in GroupKFold(n_splits=splits).split(base, groups=a_groups):
        heldout = set(a_groups.iloc[test])
        train = np.flatnonzero(~b_groups.isin(heldout).to_numpy())
        assert not heldout.intersection(set(b_groups.iloc[train]))
        pairs.append((train, np.asarray(test, dtype=int)))
    return pairs


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(prediction) - np.asarray(y)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "bias": float(residual.mean()),
        "p90_ae": float(np.quantile(abs(residual), 0.9)),
    }


def _frame(raw: pd.DataFrame) -> pd.DataFrame:
    return canonicalize_coordinates(
        normalize_missing_values(clean_detection_limits(raw))
    )


def run_experiments(
    root: Path = PROJECT_ROOT,
    destination: Path | None = None,
    *,
    folds: int = 5,
) -> dict[str, Any]:
    """Development comparison only: A test groups fixed for all candidate training."""
    out = destination or (
        RESULTS_DIR / "scientific_simulations" / "vnext_20261106" / "stage2_m1"
    )
    out.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(
        root / "data/processed/model_1/training_combined.csv", low_memory=False
    )
    original = _frame(original).dropna(subset=["Li_icpms"])
    original_mask = protocol_masks(original)["A_original"]
    a = original.loc[original_mask].copy().reset_index(drop=True)
    assert len(a) and a["source_dataset"].nunique() == 2

    raw_file = root / "data/raw/model_1/georoc/GEOROC_Andean_Arc_LithiumScope.csv"
    raw = pd.read_csv(raw_file, low_memory=False)
    georoc = harmonize_feot_candidate(raw, raw_file.name)
    mamani = original.loc[
        original["source_dataset"].eq("Mamani09 bootstrap")
    ].copy()
    merged, dedup_audit = merge_harmonized_sources([mamani, georoc])
    merged.to_csv(out / "feot_combined_experiment_only.csv", index=False)
    prepared = quality_candidate_frames(merged)
    b = prepared["B"].reset_index(drop=True)
    c = prepared["C"].reset_index(drop=True)
    print(
        "M1_DATASETS", {"A": len(a), "B": len(b), "C": len(c)},
        "GEOROC_MERGE", dedup_audit, flush=True,
    )

    bundle = joblib.load(
        root / "models/model_1/trained/training_20261105_204029_m0300/model.joblib"
    )
    estimator = bundle["estimator"]
    schema = bundle["schema"]
    for name, frame in [("A", a), ("B", b), ("C", c)]:
        if not frame["Li_icpms"].notna().all():
            raise ValueError("Invalid Li target in protocol " + name)
        if frame[["Longitude", "Latitude"]].isna().any().any():
            raise ValueError("Missing spatial groups in protocol " + name)

    x = {k: prepare_prediction_data(f, "svm", schema) for k, f in
         [("A", a), ("B", b), ("C", c)]}
    target = {k: pd.to_numeric(f["Li_icpms"], errors="coerce").to_numpy()
              for k, f in [("A", a), ("B", b), ("C", c)]}
    a_group = build_spatial_groups(a)
    if a_group is None:
        raise ValueError("No A spatial groups")
    test_folds = [
        np.asarray(test, dtype=int)
        for _, test in GroupKFold(n_splits=folds).split(a, groups=a_group)
    ]
    specs = [
        ("A_identity", "A", "identity", {}),
        ("B_feot_identity", "B", "identity", {}),
        ("B_feot_log1p", "B", "log1p", {}),
        ("B_feot_regularized", "B", "identity", {
            "model__C": float(estimator.get_params()["model__C"]) / 4,
        }),
        ("C_provisional_identity", "C", "identity", {}),
    ]
    records = []
    paired = a[["Li_icpms", "source_dataset", "Longitude", "Latitude"]].copy()
    paired["fold"] = -1
    for foldnum, testidx in enumerate(test_folds, start=1):
        paired.iloc[testidx, paired.columns.get_loc("fold")] = foldnum
    assert paired.fold.ge(1).all()

    for label, version, transform, params in specs:
        fold_splits = heldout_group_masks(a, {"A": a, "B": b, "C": c}[version],
                                          splits=folds)
        pred = np.full(len(a), np.nan)
        for fold, (trainidx, testidx) in enumerate(fold_splits, start=1):
            assert np.array_equal(np.sort(testidx), np.sort(test_folds[fold - 1]))
            model = clone(estimator)
            if params:
                model.set_params(**params)
            model = wrap_target_regressor(model, transform)
            model.fit(x[version].iloc[trainidx], target[version][trainidx])
            values = np.asarray(model.predict(x["A"].iloc[testidx]), dtype=float)
            pred[testidx] = values
            m = _metrics(target["A"][testidx], values)
            records.append({
                "variant": label, "fold": fold, "n_train": len(trainidx),
                "n_test": len(testidx), "source_variant": version,
                "target_transform": transform, **m,
            })
            print("M1_FOLD",label,fold,"RMSE",round(m["rmse"],4),
                  "TRAIN",len(trainidx),flush=True)
        if not np.isfinite(pred).all():
            raise RuntimeError("Missing common test predictions for "+label)
        paired[label] = pred
    paired.to_csv(out/"m1_paired_oof.csv",index=False)
    table=pd.DataFrame(records)
    table.to_csv(out/"m1_development_fold_metrics.csv",index=False)
    summary = table.groupby("variant",sort=True).agg(
        mean_fold_mae=("mae","mean"),mean_fold_rmse=("rmse","mean"),
        mean_fold_bias=("bias","mean"),train_count=("n_train","mean"),
    ).reset_index()
    source_metrics=[]
    for label,*_ in specs:
        for name,g in paired.groupby("source_dataset",sort=True):
            source_metrics.append({
                "variant":label,"source":name,"n":len(g),
                **_metrics(g.Li_icpms.to_numpy(),g[label].to_numpy()),
            })
    pd.DataFrame(source_metrics).to_csv(out/"m1_paired_by_source.csv",index=False)
    summary.to_csv(out/"m1_development_rankings.csv",index=False)

    # An optional experimental fit artifact is not a final model, not installed,
    # and NEVER promoted based on these reused development folds.
    candidates = summary.loc[summary.variant.str.startswith("B_")].sort_values(
        "mean_fold_rmse"
    )
    choice = str(candidates.iloc[0].variant)
    spec = next((s for s in specs if s[0] == choice))
    _, source_variant, transformation, tuning = spec
    chosen=clone(estimator)
    if tuning:
        chosen.set_params(**tuning)
    chosen=wrap_target_regressor(chosen,transformation)
    chosen.fit(x[source_variant],target[source_variant])
    modeldir=out/"experimental_models"
    modeldir.mkdir(exist_ok=True)
    joblib.dump(chosen, modeldir/"model_1_development_only.joblib")
    manifest={
        "phase":"DEVELOPMENT_ONLY_NOT_RELEASE",
        "chosen_variant":choice,
        "candidate_training_size":len({"A":a,"B":b,"C":c}[source_variant]),
        "training_original_baseline_size":len(a),
        "feot_conversion":IRON_FEOT_TO_FE2O3T,
        "source_hash":hashlib.sha256(raw_file.read_bytes()).hexdigest(),
        "source_comparison":"Common test indices from original QC A, group disjoint",
        "data_acquisition":"NONE",
        "c_rows_uncertified":True,
        "ranking":summary.to_dict("records"),
        "source_metrics":source_metrics,
        "dedup":dedup_audit,
        "model_is_in_experimental_results_only":True,
    }
    (out/"m1_experiment_manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2,default=str),
        encoding="utf8",
    )
    return manifest


if __name__ == "__main__":
    report=run_experiments()
    print("M1_DEVELOPMENT_DONE",report["chosen_variant"],flush=True)
