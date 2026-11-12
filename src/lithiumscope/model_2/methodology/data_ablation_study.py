"""Three-arm train-only M2 data-quality ablation on frozen spatial folds.

Controlled alternatives: unchanged QA95 train, omit provisional GEOROC only
from TRAIN, omit historically conflicting-site observations only from TRAIN.
Every held-out observation is retained. No permanent source modification,
model promotion, random holdout, or PDF/menu work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.regularization_study import _load_inputs


def _source_flags(audit: pd.DataFrame) -> dict[str, np.ndarray]:
    source = audit.source_cohort.astype(str).str.lower()
    mixed = audit.mixed_label_site_at_Li21p6.to_numpy(dtype=bool)
    qa = audit.train_quality_pass_95.to_numpy(dtype=bool)
    return {
        "all_qa95_train": qa,
        "historical_only_qa95_train": qa & (source == "historical").to_numpy(),
        "conflicting_sites_excluded_from_train": qa & ~mixed,
    }


def run_study(*, root: Path = PROJECT_ROOT, output: Path) -> dict:
    manifest, x, audit, bundle, folds = _load_inputs(root)
    masks = _source_flags(audit)
    if not (masks["all_qa95_train"].sum() == 767
            and masks["historical_only_qa95_train"].sum() < 767
            and masks["conflicting_sites_excluded_from_train"].sum() < 767):
        raise ValueError("Frozen science audit cohorts changed")
    scores = []
    metrics = []
    for k, fold in enumerate(folds, start=1):
        test = fold.test_idx
        y_test = np.asarray(fold.y_test, dtype=int)
        for arm, mask in masks.items():
            train = fold.train_idx[mask[fold.train_idx]]
            y_train = (
                manifest.Li_icpms.iloc[train].to_numpy(float)
                >= fold.label_threshold_ppm
            ).astype(int)
            if len(np.unique(y_train)) != 2:
                raise ValueError(f"Missing training class in {arm}, fold {k}")
            train_groups = set(manifest.spatial_group.iloc[train])
            test_groups = set(manifest.spatial_group.iloc[test])
            if train_groups & test_groups:
                raise ValueError("Spatial leakage in controlled ablation")
            model = clone(bundle["estimator"]).set_params(random_state=42 + k)
            model.fit(x.iloc[train], y_train)
            prob = model.predict_proba(x.iloc[test])[:, 1]
            metrics.append({
                "arm": arm, "fold": k, "n_train": len(train),
                "n_test": len(test),
                "roc_auc": float(roc_auc_score(y_test, prob)),
                "average_precision": float(average_precision_score(y_test, prob)),
                "test_groups": len(test_groups),
            })
            for i, row in enumerate(test):
                scores.append({
                    "arm": arm, "fold": k,
                    "sample_index": int(row),
                    "sample_id": manifest.sample_id.iloc[row],
                    "spatial_group": manifest.spatial_group.iloc[row],
                    "source_cohort": audit.source_cohort.iloc[row],
                    "y_true": int(y_test[i]),
                    "score": float(prob[i]),
                })
        print(f"Data ablation outer fold {k}/5 completed", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame(metrics)
    cases = pd.DataFrame(scores)
    per_fold.to_csv(output / "fold_metrics.csv", index=False)
    cases.to_csv(output / "paired_outer_predictions.csv", index=False)
    report = {}
    for arm, group in cases.groupby("arm"):
        assert len(group) == 783
        assert group.sample_index.nunique() == 783
        report[arm] = {
            "outer_rows": int(len(group)),
            "roc_auc_pooled": float(roc_auc_score(group.y_true, group.score)),
            "average_precision_pooled": float(
                average_precision_score(group.y_true, group.score)
            ),
            "roc_auc_fold_mean": float(
                per_fold.loc[per_fold.arm == arm, "roc_auc"].mean()
            ),
            "train_eligible_full_dataset": int(masks[arm].sum()),
        }
    result = {
        "state": "exploratory_not_release_validated",
        "original_rows_preserved": 783,
        "historical_folds_preserved": 706,
        "outer_cases_never_filtered": True,
        "arms": report,
        "external_demo_used_to_select_arms": False,
        "purpose": (
            "Isolate provisional GEOROC and conflicting historical-site "
            "influences on spatial generalization, not a new release."
        ),
    }
    (output / "data_ablation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_study(output=args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
