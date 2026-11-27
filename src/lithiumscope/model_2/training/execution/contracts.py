"""Typed data and evidence exchanged between M2 competition components."""
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd
from lithiumscope.model_2.validation.spatial import ScientificFold


@dataclass
class CompetitionOutcome:
    run_dir: Path
    ranking: pd.DataFrame
    winner: str | None
    successful: list[str]
    failed: list[str]


class ScientificData(NamedTuple):
    fold_ids: pd.Series
    frame: pd.DataFrame
    groups: pd.Series | None
    quality_audit: dict
    quality_cfg: dict
    quality_mask: np.ndarray | None
    scientific_folds: list[ScientificFold]
    threshold: float
    x: pd.DataFrame
    y: pd.Series
    y_full: pd.Series


class ScientificManifests(NamedTuple):
    baseline: tuple
    dataset_manifest: object
    dataset_manifest_path: Path
    folds: int


@dataclass(frozen=True)
class CompetitionOptions:
    fixed_threshold: float | None
    inner_folds: int
    lithium_column: str
    manifest_path: Path
    optimization_cfg: dict
    q: float
    seed: int


@dataclass
class CandidateEvidence:
    results: dict
    rows: list[dict]
    failed: list[str]
