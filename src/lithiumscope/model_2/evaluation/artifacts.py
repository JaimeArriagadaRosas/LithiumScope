"""Workbook evidence for the full M2 competition."""
from pathlib import Path
import pandas as pd


def competition_workbook_sheets(
    ranking: pd.DataFrame,
    results: dict[str, object],
    y: pd.Series,
    baseline,
    evidence_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {
        "ranking": ranking,
        "baseline_folds": baseline[1],
    }
    for algorithm, result in results.items():
        sheets[f"folds_{algorithm}"[:31]] = result.fold_table
        sheets[f"pred_{algorithm}"[:31]] = pd.DataFrame(
            {
                "y_true": y.to_numpy(),
                "prospectivity_score": result.probabilities,
                "predicted_class_0_5": (result.probabilities >= 0.5).astype(int),
            }
        )
    if evidence_dir is not None:
        for filename,sheet in (
            ("old_release_protocol_comparison_706.csv","release_706"),
            ("old_release_protocol_intervals_706.csv","release_intervals"),
            ("scientific_stratified_metrics.csv","scientific_strata"),
            ("three_zone_competition_cases.csv","three_zone_cases"),
        ):
            evidence_path=evidence_dir/filename
            if evidence_path.exists():
                sheets[sheet]=pd.read_csv(evidence_path)
    return sheets
