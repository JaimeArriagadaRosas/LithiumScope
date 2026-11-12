"""Scientific M2 release gate; never triggers installation or model promotion.

A release must use an independent, unconsulted site/source holdout. Internal
OOF, the ten-case demo, and proposed duplicate samples cannot certify one.
Thresholds are declared *before* reading a future holdout's model predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ExternalGatePolicy:
    min_samples: int = 100
    min_positive: int = 20
    min_negative: int = 20
    min_geographic_regions: int = 3
    min_auc: float = 0.60
    min_ap_gain: float = -0.02
    min_auc_gain: float = 0.0
    max_brier_regression: float = 0.02
    min_auc_gain_ci_lower: float = -0.02

    def validate(self) -> None:
        if any(
            number < 1 for number in (
                self.min_samples,
                self.min_positive,
                self.min_negative,
                self.min_geographic_regions,
            )
        ):
            raise ValueError("External validation support must be positive")
        if not (0 <= self.min_auc <= 1):
            raise ValueError("Invalid minimum ROC-AUC")


def _finite_scalar(value, name: str) -> float:
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"Missing/nonfinite release metric: {name}")
    return float(value)


def assess_release(
    evidence: dict,
    *,
    policy: ExternalGatePolicy = ExternalGatePolicy(),
) -> dict:
    """Return a nonpromoting, auditable approval assessment.

    Expected fields are only accepted from pre-registered, independent
    external sources. The caller has to supply documentary verification of
    independence; a lack of close coordinate matches is not enough.
    """
    policy.validate()
    violations = []
    if not evidence.get("external_independence_verified") is True:
        violations.append("independent_source_and_site_holdout_not_verified")
    if not evidence.get("holdout_unconsulted_during_model_selection") is True:
        violations.append("holdout_reused_for_model_or_threshold_selection")
    if not evidence.get("sample_identity_and_provenance_verified") is True:
        violations.append("sample_provenance_not_verified")
    if evidence.get("possible_duplicate_count", 0):
        violations.append("possible_training_or_holdout_duplicates")
    if evidence.get("labels_field_verified") is not True:
        violations.append("external_lithium_labels_not_certified")
    if evidence.get("regions", 0) < policy.min_geographic_regions:
        violations.append("insufficient_independent_geographic_regions")
    if evidence.get("n", 0) < policy.min_samples:
        violations.append("insufficient_external_samples")
    if evidence.get("positives", 0) < policy.min_positive:
        violations.append("insufficient_external_positives")
    if evidence.get("negatives", 0) < policy.min_negative:
        violations.append("insufficient_external_negatives")

    # No numeric pass is possible from a missing or invalid metric.
    metrics = (
        "candidate_auc", "reference_auc", "candidate_ap", "reference_ap",
        "candidate_brier", "reference_brier", "paired_auc_gain_ci_lower",
    )
    values = {}
    for key in metrics:
        try:
            values[key] = _finite_scalar(evidence.get(key), key)
        except (TypeError, ValueError):
            violations.append(f"missing_valid_{key}")
    if len(values) == len(metrics):
        if values["candidate_auc"] < policy.min_auc:
            violations.append("absolute_discrimination_below_minimum")
        if (values["candidate_auc"] - values["reference_auc"]
                < policy.min_auc_gain):
            violations.append("candidate_auc_below_reference")
        if (values["candidate_ap"] - values["reference_ap"]
                < policy.min_ap_gain):
            violations.append("candidate_ap_material_regression")
        if (values["candidate_brier"] - values["reference_brier"]
                > policy.max_brier_regression):
            violations.append("candidate_calibration_material_regression")
        if values["paired_auc_gain_ci_lower"] < policy.min_auc_gain_ci_lower:
            violations.append("spatial_paired_gain_too_uncertain")

    approved_for_scientific_review = not violations
    return {
        "gate_version": 1,
        "scientific_release_evidence_pass": approved_for_scientific_review,
        "automatic_model_activation": False,
        "requires_explicit_human_approval": True,
        "violations": violations,
        "policy": asdict(policy),
        "evidence": evidence,
        "interpretation": (
            "Evidence passes predeclared gate; publication still requires approval"
            if approved_for_scientific_review
            else "Candidate remains experimental, not eligible for release"
        ),
    }
