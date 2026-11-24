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
            type(number) is not int or number < 1 for number in (
                self.min_samples,
                self.min_positive,
                self.min_negative,
                self.min_geographic_regions,
            )
        ):
            raise ValueError("External validation support must be positive")
        if not (0 <= self.min_auc <= 1 and 0 <= self.max_brier_regression <= 1):
            raise ValueError("Invalid absolute metric policy")
        if not all(np.isfinite(v) and -1<=v<=1 for v in (self.min_ap_gain,self.min_auc_gain,self.min_auc_gain_ci_lower)):
            raise ValueError("Invalid paired gain policy")


def _finite_scalar(value, name: str) -> float:
    if value is None or isinstance(value,(bool,np.bool_)) or not np.isfinite(float(value)):
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
    duplicate_count=evidence.get("possible_duplicate_count")
    if type(duplicate_count) is not int or duplicate_count!=0:
        violations.append("possible_training_or_holdout_duplicates")
    if evidence.get("labels_field_verified") is not True:
        violations.append("external_lithium_labels_not_certified")
    counts={}
    for name,minimum,reason in (
        ("regions",policy.min_geographic_regions,"insufficient_independent_geographic_regions"),
        ("n",policy.min_samples,"insufficient_external_samples"),
        ("positives",policy.min_positive,"insufficient_external_positives"),
        ("negatives",policy.min_negative,"insufficient_external_negatives")):
        value=evidence.get(name)
        if type(value) is not int or value<minimum:
            violations.append(reason)
        if type(value) is int and value>=0:
            counts[name]=value
    if len(counts)!=4 or counts["positives"]+counts["negatives"]!=counts["n"] or counts["regions"]>counts["n"]:
        violations.append("inconsistent_external_support")

    # No numeric pass is possible from a missing or invalid metric.
    metrics = (
        "candidate_auc", "reference_auc", "candidate_ap", "reference_ap",
        "candidate_brier", "reference_brier", "paired_auc_gain_ci_lower",
    )
    values = {}
    for key in metrics:
        try:
            value = _finite_scalar(evidence.get(key), key)
            lower=-1 if key=="paired_auc_gain_ci_lower" else 0
            if not lower<=value<=1:
                raise ValueError("metric outside domain")
            values[key]=value
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
