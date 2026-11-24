"""Spatial training guard: unresolved aliases or CRS cannot certify a spatial split."""
from __future__ import annotations


def coordinate_preflight(prepared,run_context,*,require_groups,outer_folds):
    coordinate=dict(prepared.coordinate_report)
    errors=[]
    warnings=[]
    if require_groups and coordinate.get("unresolved_coordinate_rows",0):
        errors.append("Retained samples have unresolved coordinate aliases, axis or CRS evidence.")
    if coordinate.get("unknown_crs_rows",0):
        warnings.append("Coordinate datum/accuracy not certified for all retained observations; spatial groups are based on reported numeric coordinates only.")
    evidence=getattr(prepared,"coordinate_evidence",None)
    if evidence is not None:
        evidence.to_csv(run_context.tables/"coordinate_alias_audit.csv",index=False)
    spatial_groups=int(prepared.groups.nunique()) if prepared.groups is not None else 0
    if require_groups:
        if prepared.groups is None:
            errors.append(
                "Spatial validation is mandatory but complete valid coordinates "
                "are not available for every retained sample. "
                f"valid={coordinate.get('valid_rows',0)} "
                f"missing={coordinate.get('missing_rows',0)} "
                f"out_of_range={coordinate.get('out_of_range_rows',0)}."
            )
        elif spatial_groups<outer_folds:
            errors.append(f"Spatial validation is mandatory but only {spatial_groups} groups are available for {outer_folds} folds.")
    return coordinate,spatial_groups,errors,warnings
