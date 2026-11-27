"""Candidate ranking and deterministic winner selection."""
import pandas as pd
from lithiumscope.core.scientific.candidate_evidence import eligible_rows


def ranking_row(result) -> dict:
    folds = result.fold_table
    return {
        "algorithm": result.algorithm,
        "label": result.label,
        "roc_auc_mean": float(folds["roc_auc"].mean()),
        "roc_auc_std": float(folds["roc_auc"].std(ddof=0)),
        "average_precision_mean": float(folds["average_precision"].mean()),
        "balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
        "f1_mean": float(folds["f1"].mean()),
        "precision_mean": float(folds["precision"].mean()),
        "recall_mean": float(folds["recall"].mean()),
        "brier_score_mean": float(folds["brier_score"].mean()),
        "elapsed_seconds": result.elapsed_seconds,
        "status": "ok",
        "is_baseline": False,
    }


def rank_candidates(rows):
    ranking = pd.DataFrame(rows)
    candidates = ranking[
        eligible_rows(ranking)
        & (ranking["is_baseline"] != True)  # noqa: E712
    ].copy()
    baseline_rows = ranking[ranking["is_baseline"] == True].copy()  # noqa: E712
    failed_rows = ranking[(~eligible_rows(ranking)) & (ranking["is_baseline"] != True)].copy()

    if not candidates.empty:
        candidates = candidates.sort_values(
            [
                "roc_auc_mean",
                "average_precision_mean",
                "balanced_accuracy_mean",
                "algorithm",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        candidates.insert(0, "rank", range(1, len(candidates) + 1))
    if not baseline_rows.empty:
        baseline_rows.insert(0, "rank", pd.NA)

    ranking = pd.concat(
        [candidates, baseline_rows, failed_rows],
        ignore_index=True,
        sort=False,
    )
    return ranking, candidates
