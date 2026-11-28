"""Approved, version-bound scientific dispositions; preserve all historical evidence."""
from ..source_studies import source_catalog

DECISION_REFERENCE = "M1 final data refinement, 28/11/2026: approved preparation plan; M1_FUENTES_Y_CIERRE.md and M1_CIERRE_DATOS_RESULTADOS.md"
DECISIONS = {
    "georoc_andean_202609": ("restricted_recovery", "Close mass expansion; retain verified metadata and measurement recovery. Six screened rows, none above 696 ppm."),
    "georoc_felsic_rocks": ("reconcile_only", "Prior transfer stability guards failed. Reconcile 69 high-Li candidates; do not repeat broad transfer."),
    "chen_global_arcs_2020": ("rejected", "Prior bounded transfer reduced academic mean RMSE 7.074911 to 7.053262 but worsened tail RMSE 24.888575 to 25.093647. Only four compatible unique rows above 40 ppm, none above 60."),
    "meixner_2020": ("rejected", "Only SiO2 among 22 canonical chemical predictors; unresolved identity, no verified site+silica match. Standards and leaching not suitable labels."),
    "fernandez_turiel_2020": ("rejected", "Six well-traced whole-rock rows, none compatible in prior five-fold chemical preflight and none above 40 ppm."),
    "macusani_2025": ("quarantine", "Publication 30009 shared with historical GEOROC; potential duplicate identities. Restrict to provenance and measurement review."),
}


def apply_decisions(profiles, issues, catalog):
    rows = []
    for key, (decision, reason) in DECISIONS.items():
        row = {"source": key, "disposition": decision, "reason": reason,
               "decision_reference": DECISION_REFERENCE, "approved_rows": 0}
        try:
            if key in issues or key not in profiles:
                raise ValueError(issues.get(key, "Profile missing"))
            profile = profiles[key]
            record = source_catalog.retirement(key, root=catalog)
            row["profile_manifest_sha256"] = profile["record"]["manifest_sha256"]
            row["source_sha256"] = profile["manifest"]["source_sha256"]
            row["profile_summary"] = profile["manifest"]["summary"]
            row["historical_evidence"] = profile["history"]
            if decision == "rejected":
                if record is not None:
                    if (record["profile_manifest_sha256"] != row["profile_manifest_sha256"]
                            or record["source_sha256"] != row["source_sha256"]):
                        raise ValueError("Existing retirement differs from approved version")
                    row["state"] = "already_retired"
                elif any((catalog / "history" / key).glob("*.json")):
                    # A manually reopened source must not be retired again by this old approval.
                    row["state"] = "reopened_requires_review"
                else:
                    source_catalog.retire(key, profile["path"], reason=reason,
                        decision_reference=DECISION_REFERENCE, all_uses_closed=True, root=catalog,
                        historical_evidence=profile["history"])
                    row["state"] = "retired"
            elif record is not None:
                row["state"] = "existing_retirement_requires_review"
            else:
                row["state"] = "restricted"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row.update(state="blocked", issue=str(exc))
        rows.append(row)
    return rows
