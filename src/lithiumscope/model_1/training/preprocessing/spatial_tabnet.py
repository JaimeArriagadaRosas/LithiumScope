"""TabNet early stopping on raw, disjoint spatial groups; refit on all allowed TRAIN."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.scientific_checks import assert_group_isolation

logger = get_logger("model_1.spatial_tabnet")


def early_stopping_partition(x, y, groups, *, seed, validation_fraction):
    if groups is None:
        raise ValueError("TabNet requires explicit spatial TRAIN groups")
    groups = groups if isinstance(groups, pd.Series) else pd.Series(groups, index=x.index)
    if not groups.index.equals(x.index) or len(y) != len(x) or groups.isna().any():
        raise ValueError("TabNet TRAIN groups/observations must be aligned")
    if groups.astype(str).str.strip().eq("").any() or groups.nunique() < 2:
        raise ValueError("TabNet requires at least two nonempty spatial groups")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("Invalid TabNet validation fraction")
    # Deterministic candidates: choose by size alone, never by target or validation loss.
    splitter = GroupShuffleSplit(n_splits=32, test_size=validation_fraction, random_state=seed)
    feasible = [(tr, va) for tr, va in splitter.split(x, y, groups)
                if len(tr) >= 2 and len(va) >= 2]
    if not feasible:
        raise ValueError("No feasible spatial early-stopping split")
    tr, va = min(feasible, key=lambda split: abs(len(split[1])/len(x)-validation_fraction))
    assert_group_isolation(tr, va, groups)
    return tr, va, groups


class SpatialTabNetPipeline(Pipeline):
    """Fitted named_steps stay compatible with normal M1 inference and importance."""
    def fit(self, X, y=None, *, groups=None, **fit_params):
        if fit_params:
            raise ValueError("Unsupported TabNet fit parameters")
        if not isinstance(X, pd.DataFrame) or not X.index.is_unique:
            raise ValueError("TabNet requires indexed raw predictor rows")
        if isinstance(y,pd.Series) and not y.index.equals(X.index):
            raise ValueError("TabNet TRAIN target index is not aligned")
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError("TabNet TRAIN target is invalid")
        template = self.named_steps["model"]
        tr, va, groups = early_stopping_partition(
            X, y, groups, seed=template.seed, validation_fraction=template.validation_fraction)
        # The early-stopping validation rows must not determine the imputer or encoder.
        prep = clone(self.named_steps["preprocess"])
        fit_x = prep.fit_transform(X.iloc[tr], y[tr])
        val_x = prep.transform(X.iloc[va])
        selector = clone(template)
        best_epoch = selector.fit_explicit_validation(fit_x, y[tr], val_x, y[va])
        # Once epoch count is fixed, refit preprocessing and model on all allowed TRAIN.
        final_prep = clone(self.named_steps["preprocess"])
        final_x = final_prep.fit_transform(X, y)
        model = clone(template)
        model.fit_fixed_epochs(final_x, y, best_epoch)
        self.steps = [("preprocess", final_prep), ("model", model)]
        self.early_stopping_evidence_ = {
            "policy": "spatial_train_only_refit_v1", "best_epoch": int(best_epoch),
            "n_train": len(tr), "n_validation": len(va),
            "train_groups": sorted(groups.iloc[tr].astype(str).unique()),
            "validation_groups": sorted(groups.iloc[va].astype(str).unique()),
            "train_positions": tr.tolist(), "validation_positions": va.tolist(),
            "refit_n": len(X),
        }
        logger.info("TabNet spatial early stopping: train=%d validation=%d epoch=%d refit=%d",
                    len(tr), len(va), best_epoch, len(X))
        return self


def fit_regression_estimator(estimator, x, y, groups=None):
    """Pass groups only to the new TabNet path, including transformed targets."""
    from lithiumscope.resources.manager import estimator_thread_scope
    raw = getattr(estimator, "regressor", estimator)
    with estimator_thread_scope():
        if getattr(estimator,"requires_spatial_groups",False):
            return estimator.fit(x,y,groups=groups)
        if isinstance(raw, SpatialTabNetPipeline):
            return estimator.fit(x, y, groups=groups)
        return estimator.fit(x, y)


def early_stopping_evidence(estimator):
    fitted = getattr(estimator, "regressor_", estimator)
    return getattr(fitted, "early_stopping_evidence_", None)
