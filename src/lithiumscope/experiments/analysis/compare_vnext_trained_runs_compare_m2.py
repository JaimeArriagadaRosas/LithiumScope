"""Separated compare_m2 responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.experiments.analysis.compare_vnext_trained_runs import Path

def compare_m2(root: Path, output: Path) -> dict:
    from lithiumscope.experiments.analysis.compare_vnext_trained_runs import NEW_M2, OLD_M2, Path, check_m2_identical_protocol, classification_metrics, group_bootstrap_gain, json, load_m2_oof, np, pd
    old_run = root / "results/model_2/runs" / OLD_M2
    new_run = root / "results/model_2/runs" / NEW_M2
    old_rf = load_m2_oof(old_run, "random_forest")
    new_rf = load_m2_oof(new_run, "random_forest")
    cal_rf = load_m2_oof(new_run, "random_forest_spatial_calibrated")
    old_cat = load_m2_oof(old_run, "catboost")
    manifest = pd.read_csv(
        root / "data/processed/model_2/training_manifest.csv"
    )
    check_m2_identical_protocol(old_run, new_run, old_rf, new_rf, manifest)
    for candidate in (cal_rf, old_cat):
        for key in ("y_true", "li_ppm", "outer_fold"):
            if not np.allclose(candidate[key], old_rf[key]):
                raise ValueError(f"M2 differing labels/folds for {key}")
    y = old_rf.y_true.to_numpy(dtype=int)
    scores = {
        "previous_rf": old_rf.prospectivity_score.to_numpy(dtype=float),
        "new_rf": new_rf.prospectivity_score.to_numpy(dtype=float),
        "new_rf_spatial_calibrated":
            cal_rf.prospectivity_score.to_numpy(dtype=float),
        "previous_catboost": old_cat.prospectivity_score.to_numpy(dtype=float),
    }
    metrics = {
        label: classification_metrics(y, value)
        for label, value in scores.items()
    }
    difference = np.max(np.abs(scores["new_rf"]-scores["previous_rf"]))
    groups = manifest.spatial_group.to_numpy()
    ci_calibrated = {
        metric: group_bootstrap_gain(
            y, scores["previous_rf"], scores["new_rf_spatial_calibrated"],
            groups, metric=metric
        ) for metric in ("roc_auc", "average_precision", "brier")
    }
    paired_rows = pd.DataFrame({
        "row_index": np.arange(len(y)),
        "spatial_group": groups,
        "outer_fold": old_rf.outer_fold,
        "y_true": y,
        "Li_ppm": old_rf.li_ppm,
        **scores,
    })
    paired_rows.to_csv(output / "m2_identical_protocol_paired_oof.csv",
                       index=False)
    rows = []
    for fold, slice_ in paired_rows.groupby("outer_fold"):
        if len(np.unique(slice_.y_true)) != 2:
            raise ValueError("M2 outer fold missing positive/negative samples")
        for name in scores:
            m = classification_metrics(slice_.y_true, slice_[name])
            rows.append({"fold": int(fold), "variant": name, **m})
    pd.DataFrame(rows).to_csv(output / "m2_paired_fold_metrics.csv",
                              index=False)
    old_perf = metrics["previous_rf"]
    new_perf = metrics["new_rf"]
    discrimination_gate = bool(
        new_perf["roc_auc"] - old_perf["roc_auc"] >= .03
        and new_perf["average_precision"] - old_perf["average_precision"] >= .02
        and new_perf["brier"] - old_perf["brier"] <= .01
    )
    report = {
        "status": "no_new_discrimination_gain",
        "old_run": OLD_M2, "new_run": NEW_M2,
        "same_manifest_same_labels_same_spatial_folds": True,
        "samples": len(y),
        "spatial_groups": int(pd.Series(groups).nunique()),
        "maximum_old_new_rf_probability_delta": float(difference),
        "new_old_rf_identical_to_tolerance_1e_12": bool(difference <= 1e-12),
        "metrics_global_oof": metrics,
        "calibrated_vs_previous_rf_bootstrap": ci_calibrated,
        "development_discrimination_gate": {
            "rules": (
                "At least +0.03 ROC-AUC and +0.02 AP relative to prior winner, "
                "without more than +0.01 degradation in Brier."
            ),
            "passes": discrimination_gate,
            "release_certified": False,
        },
        "assessment": (
            "The RF winner is numerically unchanged; the new calibrated "
            "variant improves Brier score but reduces ROC-AUC/AP and at 0.5 "
            "has no sensitivity. No improvement in discriminative accuracy "
            "has been demonstrated over previous training."
        ),
        "release_limit": (
            "No untouched external Li-labelled holdout. Internal OOF cannot "
            "alone authorize promotion; active September release was not "
            "measured here under the same protocol."
        ),
    }
    (output / "m2_paired_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("M2_SAVED_COMPARISON", json.dumps({
        "max_probability_delta": difference,
        "metrics": metrics,
        "CI": ci_calibrated,
    }, ensure_ascii=False), flush=True)
    return report
