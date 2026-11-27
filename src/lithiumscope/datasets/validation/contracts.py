"""Explicit, opt-in dataset admission policies; frozen cohorts are never rewritten."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAJOR_OXIDES = ("SiO2", "TiO2", "Al2O3", "Fe2O3", "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5")


@dataclass(frozen=True)
class QualityPolicy:
    model_group: str
    target: str = "Li_icpms"
    required_columns: tuple[str, ...] = ()
    zero_policy: str = "retain"
    require_coordinates: bool = True
    require_reported_crs: bool = False
    require_reported_precision: bool = False
    max_uncertainty_m: float | None = None
    oxide_columns: tuple[str, ...] = ()
    oxide_total_bounds: tuple[float, float] | None = None
    minimum_predictors: int = 0
    predictor_columns: tuple[str, ...] = ()
    unit_column: str | None = None
    method_column: str | None = None
    image_column: str | None = None
    review_censored_target: bool = False
    review_zero_target: bool = False
    identity_columns: tuple[str, ...] = ()

    def __post_init__(self):
        if self.model_group not in {"model_1", "model_2"}:
            raise ValueError("Unknown model group")
        if self.zero_policy not in {"retain", "exclude"}:
            raise ValueError("Unknown zero policy")
        if not self.target or self.minimum_predictors < 0 or self.minimum_predictors > len(set(self.predictor_columns)):
            raise ValueError("Invalid target or predictor policy")
        if self.target in self.predictor_columns or len(set(self.predictor_columns)) != len(self.predictor_columns):
            raise ValueError("Predictor policy contains target or duplicates")
        if self.max_uncertainty_m is not None and (not np.isfinite(self.max_uncertainty_m) or self.max_uncertainty_m < 0):
            raise ValueError("Invalid maximum uncertainty")
        if len(set(self.identity_columns)) != len(self.identity_columns):
            raise ValueError("Duplicate identity columns in policy")
        if self.oxide_total_bounds is not None and not self.oxide_columns:
            raise ValueError("Oxide total policy requires oxide columns")


def candidate_policy(model_group: str, **overrides) -> QualityPolicy:
    """Conservative candidate checks, not a replacement for a frozen protocol."""
    if model_group == "model_1":
        defaults = {"oxide_columns": MAJOR_OXIDES, "unit_column": "source_li_unit",
                    "method_column": "source_li_method"}
    elif model_group == "model_2":
        defaults = {"image_column": "image_path", "require_reported_precision": True,
                    "require_reported_crs": True, "unit_column": "source_li_unit"}
    else:
        raise ValueError("Unknown model group")
    return QualityPolicy(model_group, **{"review_censored_target": True, "review_zero_target": True,
                                         **defaults, **overrides})
