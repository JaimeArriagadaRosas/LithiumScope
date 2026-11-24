"""Predeclared feature ablations; no selection using the outer TEST."""
from __future__ import annotations

from dataclasses import replace

from lithiumscope.model_1.training.preprocessing.tabular import (
    SPATIAL_CANDIDATES, TRACE_ELEMENTS, DERIVED, FeatureSchema,
)

ABLATIONS={
    "reference":set(),
    "without_coordinates":set(SPATIAL_CANDIDATES),
    "without_regional_context":set(SPATIAL_CANDIDATES+["Arc","Domain"]),
    "without_trace_elements":set(TRACE_ELEMENTS),
    "without_derived":set(DERIVED),
}


def feature_variant(prepared, name, *, preprocessing_policy=None):
    if name not in ABLATIONS:
        raise ValueError("Unknown predeclared feature variant")
    excluded=ABLATIONS[name]
    base=prepared.schema
    schema=FeatureSchema([c for c in base.numeric if c not in excluded],
                         [c for c in base.categorical if c not in excluded],
                         base.preparation_version,
                         preprocessing_policy or getattr(base,"preprocessing_policy","legacy"))
    columns=schema.numeric+schema.categorical
    if not columns or not set(columns)<=set(prepared.x.columns):
        raise ValueError("Invalid feature ablation")
    return replace(prepared,schema=schema,x=prepared.x.loc[:,columns].copy())
