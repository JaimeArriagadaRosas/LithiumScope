"""Paired development diagnostics and limited group-heldout learning curve M1."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from lithiumscope.persistence.model_deserialization import load_model_artifact
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.model_1.methodology.dataset_comparison import (
    _frame, _metrics, quality_candidate_frames,
)
from lithiumscope.model_1.pipeline import prepare_prediction_data
from lithiumscope.model_1.validation.spatial import build_spatial_groups


from lithiumscope.model_1.methodology.dataset_comparison_diagnostics import (
    cluster_bootstrap_improvement as cluster_bootstrap_improvement,
)


def run_diagnostics(
    root: Path = PROJECT_ROOT, dest: Path | None = None,
) -> dict:
    out = dest or (
        RESULTS_DIR / "scientific_simulations" / "vnext_20261106" / "stage2_m1"
    )
    paired = pd.read_csv(out / "m1_paired_oof.csv")
    groups = build_spatial_groups(paired)
    if groups is None:
        raise RuntimeError("No spatial groups for paired OOF")
    y = paired.Li_icpms.to_numpy(dtype=float)
    b = paired.B_feot_identity.to_numpy(dtype=float)
    a = paired.A_identity.to_numpy(dtype=float)
    ci = cluster_bootstrap_improvement(y, a, b, groups)
    quantiles = pd.qcut(paired.Li_icpms, 4, duplicates="drop")
    rows = []
    for groupname, frame in paired.groupby(quantiles, observed=True):
        for variant in ("A_identity", "B_feot_identity", "C_provisional_identity"):
            rows.append({
                "lithium_band": str(groupname), "n": len(frame),
                "variant": variant, "min_li": float(frame.Li_icpms.min()),
                "max_li": float(frame.Li_icpms.max()),
                **_metrics(frame.Li_icpms.to_numpy(), frame[variant].to_numpy()),
            })
    pd.DataFrame(rows).to_csv(out / "m1_paired_by_lithium.csv", index=False)

    # Learning curve: one fixed heldout fold, random subsets of *train groups*.
    combined = pd.read_csv(out / "feot_combined_experiment_only.csv", low_memory=False)
    versions = quality_candidate_frames(_frame(combined))
    b_df = versions["B"].reset_index(drop=True)
    bundle = load_model_artifact(
        root / "models/model_1/trained/training_20261105_204029_m0300/model.joblib", model_group="model_1"
    )
    b_groups = build_spatial_groups(b_df)
    test_fold = int(paired.fold.min())
    test_indices = np.flatnonzero(paired.fold.to_numpy() == test_fold)
    heldout = set(groups.iloc[test_indices])
    allowed = ~b_groups.isin(heldout)
    train_groups = np.asarray(sorted(b_groups.loc[allowed].unique()))
    rng = np.random.default_rng(42)
    ordered_groups = rng.permutation(train_groups)
    x_train = prepare_prediction_data(b_df, "svm", bundle["schema"])
    x_test = prepare_prediction_data(
        pd.read_csv(
            root / "data/processed/model_1/training_combined.csv", low_memory=False
        ).pipe(_frame).dropna(subset=["Li_icpms"]).loc[
            lambda frame: pd.to_numeric(frame["SUM (no water)"], errors="coerce")
            .between(94, 102)
        ].reset_index(drop=True).iloc[test_indices], "svm", bundle["schema"]
    )
    target = pd.to_numeric(b_df.Li_icpms, errors="coerce").to_numpy()
    learning = []
    for fraction in (0.30, 0.60, 1.0):
        n_groups = max(5, int(round(fraction * len(train_groups))))
        chosen = set(ordered_groups[:n_groups])
        idx = np.flatnonzero(allowed.to_numpy() & b_groups.isin(chosen).to_numpy())
        est = clone(bundle["estimator"])
        est.fit(x_train.iloc[idx], target[idx])
        predicted = est.predict(x_test)
        report = _metrics(y[test_indices], np.asarray(predicted, dtype=float))
        learning.append({
            "fraction_of_training_groups": fraction,
            "n_groups": n_groups, "n_train": len(idx),
            "n_test": len(test_indices), "heldout_fold": test_fold, **report,
        })
        print("M1_LEARNING_CURVE", learning[-1], flush=True)
    pd.DataFrame(learning).to_csv(out / "m1_group_learning_curve.csv", index=False)
    report = {
        "scope": "development_grouped_cv_no_independent_holdout",
        "paired_cluster_bootstrap": ci,
        "learning_curve": learning,
        "learning_curve_caution": "one heldout fold and group fractions; not an external test",
        "quality_B": "FeOT-derived analytical closure with GEOROC provenance",
        "quality_C": "eight additional unverified rows; no production certification",
    }
    (out / "m1_development_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    return report


if __name__ == "__main__":
    print("M1_DIAGNOSTICS",run_diagnostics(),flush=True)
