"""Predeclared treatment hypotheses for M1. No TEST-dependent fitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import PowerTransformer


ALGORITHMS = ("random_forest", "hist_gradient_boosting", "catboost")


@dataclass(frozen=True)
class Arm:
    name: str
    weight: str = "none"
    transform: str = "identity"
    resample: str = "none"
    features: str = "original"
    impute: bool = False
    correction: bool = False


ARMS = (
    Arm("control"),
    Arm("density_moderate", weight="density_05"),
    Arm("density_strong", weight="density_10"),
    Arm("source_group_balance", weight="source_group"),
    Arm("adaptive_quantile", resample="quantile"),
    Arm("identity_target"),
    Arm("power_075", transform="power075"),
    Arm("yeo_johnson", transform="yeo_johnson"),
    Arm("sqrt_crossfit_correction", transform="sqrt", correction=True),
    Arm("stable_features", features="stable"),
    Arm("geochemical_ratios", features="ratios"),
    Arm("geology_imputation", impute=True),
    Arm("constrained_smoter", resample="smoter"),
    Arm("density_power_ratios", weight="density_05", transform="power075", features="ratios"),
    Arm("balance_sqrt_stable", weight="source_group", transform="sqrt", features="stable", correction=True),
)


def weights(y, groups, sources, mode):
    """Density on log1p TRAIN target; bounded weights protect against domination."""
    y = np.asarray(y, dtype=float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid TRAIN target")
    if mode == "none":
        return np.ones(len(y))
    if mode == "source_group":
        frame = pd.DataFrame({"source": np.asarray(sources), "group": np.asarray(groups)})
        sizes = frame.groupby(["source", "group"])["source"].transform("size")
        n_groups = frame.groupby("source")["group"].transform("nunique")
        raw = 1 / np.sqrt(sizes.to_numpy(float) * n_groups.to_numpy(float))
    elif mode in {"density_05", "density_10"}:
        z = np.log1p(y).reshape(-1, 1)
        bandwidth = max(.15, 1.06 * float(z.std()) * len(z)**(-.2))
        density = KernelDensity(bandwidth=bandwidth).fit(z).score_samples(z)
        power = .5 if mode == "density_05" else 1.
        raw = np.exp(np.clip(-power * (density - np.median(density)), -10, 10))
    else:
        raise ValueError("Unknown weighting")
    raw = np.clip(raw / raw.mean(), .2, 5.)
    return raw / raw.mean()


def adaptive_positions(y, groups, seed):
    """Copy real rare TRAIN rows, balanced across quantile bins and spatial groups."""
    y = pd.Series(np.asarray(y, dtype=float))
    groups = pd.Series(np.asarray(groups))
    if len(y) != len(groups) or not len(y):
        raise ValueError("Misaligned TRAIN inputs")
    thresholds = y.quantile([.75, .9, .95, .99]).to_numpy()
    bins = np.searchsorted(thresholds, y, side="right")
    candidates = np.flatnonzero(bins > 0)
    if not len(candidates):
        return np.array([], dtype=int)
    keys = pd.Series(list(zip(bins[candidates], groups.iloc[candidates], strict=True)))
    inverse = 1 / keys.map(keys.value_counts()).to_numpy(float)
    pool = np.repeat(candidates, 4)
    probability = np.repeat(inverse, 4)
    probability /= probability.sum()
    return np.random.default_rng(seed).choice(
        pool, size=min(len(pool), max(1, len(y)//4)), replace=False, p=probability)


class Target:
    def __init__(self, mode):
        self.mode = mode

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Invalid target")
        self.maximum = float(y.max())
        if self.mode == "yeo_johnson":
            self.power = PowerTransformer(method="yeo-johnson", standardize=False).fit(y.reshape(-1, 1))
        return self

    def encode(self, y):
        y = np.asarray(y, dtype=float)
        if self.mode == "identity":
            return y
        if self.mode == "sqrt":
            return np.sqrt(y)
        if self.mode == "power075":
            return y**.75
        if self.mode == "yeo_johnson":
            return self.power.transform(y.reshape(-1, 1)).ravel()
        raise ValueError("Unknown transformation")

    def decode(self, z):
        z = np.maximum(np.asarray(z, dtype=float), 0.)
        with np.errstate(over="raise", invalid="raise"):
            if self.mode == "sqrt":
                result = z**2
            elif self.mode == "power075":
                result = z**(4/3)
            elif self.mode == "yeo_johnson":
                # Avoid the finite-domain singularity when the fitted lambda is negative.
                z = np.minimum(z, self.encode([self.maximum])[0])
                result = self.power.inverse_transform(z.reshape(-1, 1)).ravel()
            elif self.mode == "identity":
                result = z
            else:
                raise ValueError("Unknown transformation")
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite decoded prediction")
        return np.maximum(result, 0.)
