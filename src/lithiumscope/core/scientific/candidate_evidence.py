"""Separate numerical evaluation, essential evidence and optional visuals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def eligible_rows(ranking: pd.DataFrame) -> pd.Series:
    eligible = ranking.get("eligible_for_selection", pd.Series(True, index=ranking.index))
    return (ranking["status"] == "ok") & eligible.map(lambda value: pd.isna(value) or str(value).lower() == "true")


def verify_candidate(result, predictions, expected_count: int, metrics: tuple[str, ...], expected_folds=None) -> None:
    from lithiumscope.core.scientific_checks import assert_oof_complete

    assert_oof_complete(predictions, expected_count)
    folds = result.fold_table
    if folds.empty or "fold" not in folds or folds["fold"].duplicated().any():
        raise ValueError("Candidate fold evidence is empty or duplicated")
    if expected_folds is not None and len(folds) != expected_folds:
        raise ValueError("Candidate fold count differs from protocol")
    if not np.isfinite(folds[list(metrics)].to_numpy(dtype=float)).all():
        raise ValueError("Candidate contains undefined comparison metrics")
    if not all(np.isfinite(float(result.overall_metrics[m])) for m in metrics):
        raise ValueError("Candidate overall comparison metrics are undefined")


def record_evidence(row: dict, tracker, logger, algorithm: str, required, optional) -> None:
    """Retain a single training row; report each independent export failure."""
    errors = {}
    required_ok = True
    for name, action in required.items():
        try:
            action()
        except Exception as exc:
            required_ok = False
            errors[name] = str(exc)
            logger.exception("Essential candidate evidence failed: %s/%s", algorithm, name)
    for name, action in optional.items():
        try:
            action()
        except Exception as exc:
            errors[name] = str(exc)
            logger.exception("Optional candidate evidence failed: %s/%s", algorithm, name)
    row["eligible_for_selection"] = required_ok
    row["artifact_state"] = "complete" if not errors else "partial" if required_ok else "failed"
    row["artifact_errors"] = "; ".join(f"{k}: {v}" for k, v in errors.items())
    tracker.event("candidate_evidence", algorithm=algorithm, state=row["artifact_state"],
                  eligible_for_selection=required_ok, errors=errors)
