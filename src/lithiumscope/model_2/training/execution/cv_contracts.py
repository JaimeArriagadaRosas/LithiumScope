"""Inputs and outputs for the outer classification fold executor."""
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from lithiumscope.resources.detection.device import DeviceInfo
from lithiumscope.model_2.validation.spatial import ScientificFold
from lithiumscope.model_2.training.factory import AlgorithmSpec


@dataclass(frozen=True)
class OuterFoldData:
    x: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None
    lithium_values: object
    quality_training_mask: np.ndarray | None


@dataclass(frozen=True)
class OuterFoldPartition:
    folds: int
    scientific_folds: list[ScientificFold] | None
    split_groups: pd.Series | None
    split_iter: list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class OuterFoldOptions:
    algorithm: str
    device: DeviceInfo
    fixed_threshold_ppm: float | None
    inner_folds: int
    label: str
    label_quantile: float
    optimize: bool
    pruning: dict
    seed: int
    spec: AlgorithmSpec
    trials: int
    parameter_space: Callable
    calibration_mode: str
    operating_objective: str | None


@dataclass
class OuterFoldEvidence:
    probabilities: np.ndarray
    records: list[dict]
    operating_records: list[dict]
