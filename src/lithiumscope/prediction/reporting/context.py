from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReportContext:
    title: str
    run_id: str
    mode: str
    interpretation: str
    model_1_identity: dict
    model_2_identity: dict
    model_1_metrics: dict
    model_2_metrics: dict
    model_1_predictions: pd.DataFrame
    model_2_predictions: pd.DataFrame
    paired: pd.DataFrame
    correlations: pd.DataFrame
    concordance: pd.DataFrame
    training_vs_external: pd.DataFrame
    overlap_audit: dict | None
    input_info: dict | None
    model_1_diagnostics: dict | None
    model_2_diagnostics: dict | None
    figures: tuple[Path, ...]
    lithium_threshold_ppm: float | None
    model_2_classification_threshold: float | None
    runtime: dict
