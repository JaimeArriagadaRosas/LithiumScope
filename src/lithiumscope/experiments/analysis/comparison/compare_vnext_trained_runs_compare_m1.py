"""Separated compare_m1 responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.experiments.analysis.compare_vnext_trained_runs import Path

def compare_m1(root: Path, output: Path) -> dict:
    from lithiumscope.experiments.analysis.compare_vnext_trained_runs import NEW_M1, OLD_M1, Path, group_bootstrap_gain, json, load_m1_oof, np, pair_shared_m1, regression_metrics
    old_run = root / "results/model_1/runs" / OLD_M1
    new_run = root / "results/model_1/runs" / NEW_M1
    old, new = load_m1_oof(old_run), load_m1_oof(new_run)
    paired, metadata = pair_shared_m1(old, new)
    y = paired.y_true.to_numpy()
    a = paired.prediction_old.to_numpy()
    b = paired.prediction_new.to_numpy()
    threshold = float(np.quantile(y, .75))
    subgroups = {
        "all_shared": paired.index >= 0,
        "high_lithium_q75_common": paired.y_true >= threshold,
        "same_outer_fold_sensitivity":
            paired.outer_fold_old == paired.outer_fold_new,
        "changed_outer_fold_sensitivity":
            paired.outer_fold_old != paired.outer_fold_new,
    }
    for src in sorted(paired.source_dataset.unique()):
        subgroups[f"source_{src}"] = paired.source_dataset.eq(src)
    stratified = {}
    for label, subset in subgroups.items():
        rows = paired.loc[subset]
        stratified[label] = {
            "old": regression_metrics(rows.y_true, rows.prediction_old),
            "new": regression_metrics(rows.y_true, rows.prediction_new),
        }
    uncertainty = {
        metric: group_bootstrap_gain(
            y, a, b, paired.spatial_group_old, metric=metric
        ) for metric in ("rmse", "mae")
    }
    report = {
        "status": "exploratory_improvement_not_release_proof",
        "old_run": OLD_M1, "new_run": NEW_M1,
        "same_model_family": "svm__identity",
        "comparison": metadata,
        "common_high_li_threshold_ppm": threshold,
        "by_subset": stratified,
        "cluster_bootstrap": uncertainty,
        "development_improvement_gate": {
            "rules": (
                "At least 80% uniquely matched old observations; "
                "at least 5% lower paired RMSE and MAE; "
                "positive exploratory grouped bootstrap lower CI for both."
            ),
            "passes_exploratory_only": bool(
                len(paired) >= .80 * len(old)
                and stratified["all_shared"]["new"]["rmse"]
                    <= .95 * stratified["all_shared"]["old"]["rmse"]
                and stratified["all_shared"]["new"]["mae"]
                    <= .95 * stratified["all_shared"]["old"]["mae"]
                and uncertainty["rmse"]["ci95_group_lower"] > 0
                and uncertainty["mae"]["ci95_group_lower"] > 0
            ),
            "release_certified": False,
            "reason": "610 of 1004 shared records changed outer-fold assignment.",
        },
        "unpaired_full_oof_not_comparable": {
            "old": regression_metrics(
                old.y_true.to_numpy(), old.prediction.to_numpy()
            ),
            "new": regression_metrics(
                new.y_true.to_numpy(), new.prediction.to_numpy()
            ),
        },
        "decision": (
            "M1 B improves exploratory paired common-record errors. "
            "Do not compare raw full-dataset CV RMSE across different "
            "lithium distributions or claim an independent release test."
        ),
    }
    paired.to_csv(output / "m1_shared_records_paired_oof.csv", index=False)
    (output / "m1_paired_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("M1_SAVED_COMPARISON", json.dumps({
        "n_shared": len(paired),
        "same_folds": metadata["same_outer_fold"],
        "different_folds": metadata["different_outer_fold"],
        "old": stratified["all_shared"]["old"],
        "new": stratified["all_shared"]["new"],
        "CI": uncertainty,
    }, ensure_ascii=False), flush=True)
    return report
