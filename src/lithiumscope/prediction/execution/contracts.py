"""Typed contracts exchanged by prediction execution and publication."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from lithiumscope.prediction.execution.session import PredictionSession


@dataclass(frozen=True)
class IntegratedPredictionResult:
    session: PredictionSession
    model_1_predictions: pd.DataFrame
    model_2_predictions: pd.DataFrame
    paired: pd.DataFrame
    correlations: pd.DataFrame
    concordance: pd.DataFrame
    model_1_metrics: dict
    model_2_metrics: dict
    interpretation: str
    report_path: Path
    workbook_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class PredictionFrames:
    model_1_predictions: pd.DataFrame
    model_2_predictions: pd.DataFrame
    model_2_cases: pd.DataFrame
    paired: pd.DataFrame
    correlations: pd.DataFrame
    concordance: pd.DataFrame
    training_vs_external: pd.DataFrame


@dataclass(frozen=True)
class PredictionEvidence:
    model_1_metrics: dict
    model_2_metrics: dict
    model_1_identity: dict
    model_2_identity: dict
    model_1_diagnostics: dict
    model_2_diagnostics: dict
    input_info: dict
    overlap_audit: dict | None
    interpretation: str


@dataclass(frozen=True)
class PredictionThresholds:
    lithium_threshold: float
    probability_threshold: float
    model_2_predicted_positive_count: int
