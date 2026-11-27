"""Read-only quality decisions with explicit reasons and non-destructive partitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.datasets.validation.chemistry.composition import assess_composition
from lithiumscope.datasets.validation.chemistry.concentration import valid_concentration
from lithiumscope.datasets.validation.coordinates import assess_coordinate_aliases
from lithiumscope.datasets.validation.spatial.precision import assess_precision
from lithiumscope.datasets.validation.contracts import QualityPolicy

_STATES = ("admitted", "quarantine", "excluded")
_UNKNOWN = {"", "unknown", "none", "nan", "n/a", "na", "<na>", "not reported"}


@dataclass(frozen=True)
class QualityReport:
    policy: QualityPolicy
    decisions: pd.DataFrame

    def summary(self) -> dict:
        return {"policy": asdict(self.policy), "rows": len(self.decisions),
                "decisions": {state: int(self.decisions.decision.eq(state).sum()) for state in _STATES},
                "reasons": self.decisions.reason.explode().dropna().value_counts().to_dict(),
                "scientific_independence_certified": False}

    def partition(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not frame.index.equals(self.decisions.index):
            raise ValueError("Quality decisions do not align with source")
        return {state: frame.loc[self.decisions.decision.eq(state)].copy() for state in _STATES}


def assess_quality(frame: pd.DataFrame, policy: QualityPolicy) -> QualityReport:
    if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Quality assessment requires nonempty data and unique axes")
    required = {policy.target, *policy.required_columns}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    status = np.zeros(len(frame), dtype=np.int8)
    reasons = [[] for _ in range(len(frame))]

    def flag(mask, reason, severity=1):
        positions = np.flatnonzero(pd.Series(mask, index=frame.index).fillna(False).to_numpy(bool))
        for pos in positions:
            reasons[pos].append(reason)
            status[pos] = max(status[pos], severity)

    numeric, valid, _ = valid_concentration(frame[policy.target], zero_policy=policy.zero_policy)
    flag(~valid, "invalid_or_missing_target", 2)
    if policy.review_zero_target:
        flag(numeric.eq(0), "zero_semantics_review")
    if policy.review_censored_target and "source_li_raw" in frame:
        from lithiumscope.datasets.validation.chemistry.analytical import parse_analytical_value
        censored = frame.source_li_raw.map(lambda value: parse_analytical_value(value, decimal_comma=False)[1] is not None)
        flag(censored, "censored_target_review")
    if policy.require_coordinates or policy.require_reported_crs or policy.require_reported_precision or policy.max_uncertainty_m is not None:
        _, evidence = assess_coordinate_aliases(frame)
        if policy.require_coordinates:
            flag(evidence.coordinate_status.isin(["missing_coordinate", "invalid_alias"]), "invalid_or_missing_coordinates", 2)
            flag(evidence.coordinate_status.isin(["conflicting_aliases", "possible_axis_swap", "requires_crs_review"]), "coordinate_review")
        if policy.require_reported_crs:
            flag(evidence.crs_status.ne("reported_wgs84"), "unverified_crs")
        precision = assess_precision(frame, max_uncertainty_m=policy.max_uncertainty_m)
        flag(precision.precision_status.isin(["invalid_uncertainty", "bounds_review", "exceeds_policy"]), "precision_review")
        if policy.require_reported_precision:
            flag(precision.reported_uncertainty_m.isna(), "unknown_precision")
    if policy.oxide_columns:
        composition = assess_composition(frame, policy.oxide_columns, total_bounds=policy.oxide_total_bounds)
        flag(composition.composition_status.isin(["invalid_component", "total_outside_policy"]), "composition_review")
        if policy.oxide_total_bounds is not None:
            flag(composition.composition_status.eq("incomplete"), "incomplete_composition")
    if policy.minimum_predictors:
        values = frame.reindex(columns=list(policy.predictor_columns)).apply(pd.to_numeric, errors="coerce")
        flag((np.isfinite(values) & values.ge(0)).sum(axis=1).lt(policy.minimum_predictors), "insufficient_predictors", 2)
    if policy.unit_column:
        units = frame.get(policy.unit_column, pd.Series("", index=frame.index)).astype("string").fillna("").str.strip().str.casefold()
        flag(~units.isin({"ppm", "mg/kg"}), "unverified_target_unit")
    if policy.method_column:
        methods = frame.get(policy.method_column, pd.Series("", index=frame.index)).astype("string").fillna("").str.strip().str.casefold()
        flag(methods.isin(_UNKNOWN), "unknown_analytical_method")
    if policy.image_column:
        images = frame.get(policy.image_column, pd.Series("", index=frame.index)).astype("string").fillna("").str.strip().str.casefold()
        flag(images.isin(_UNKNOWN), "missing_image_reference", 2)
    decisions = pd.DataFrame({"source_position": np.arange(len(frame)), "decision": [_STATES[s] for s in status],
                              "reason": reasons}, index=frame.index)
    return QualityReport(policy, decisions)


def write_quality_report(report: QualityReport, directory: Path) -> Path:
    """Publish a single atomic JSON artifact; original rows stay untouched."""
    from lithiumscope.core.logging.session import get_logger
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "quality_report.json"
    temporary = directory / f".quality_report.{uuid4().hex}.tmp"
    payload = {**report.summary(), "row_decisions": report.decisions.to_dict(orient="records")}
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    get_logger("datasets.quality").info("Quality report %s: %s", destination, payload["decisions"])
    return destination
