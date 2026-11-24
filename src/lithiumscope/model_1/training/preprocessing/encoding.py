"""Fold-local explicit missing and unseen categorical encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


class ExplicitCategoryEncoder(TransformerMixin, BaseEstimator):
    MISSING = "__lithiumscope_missing__"
    UNSEEN = "__lithiumscope_unseen__"

    def _array(self, X):
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2:
            raise ValueError("Expected categorical matrix")
        return arr

    def _encode(self, X, *, fitting=False):
        arr = self._array(X)
        result = np.empty(arr.shape, dtype=object)
        if not fitting and arr.shape[1] != self.n_features_in_:
            raise ValueError("Categorical feature count changed")
        for i in range(arr.shape[1]):
            for j, value in enumerate(arr[:, i]):
                if pd.isna(value):
                    result[j, i] = self.MISSING
                else:
                    text = str(value)
                    if text in {self.MISSING, self.UNSEEN}:
                        raise ValueError("Reserved categorical token in raw input")
                    result[j, i] = text if fitting or text in self.vocabulary_[i] else self.UNSEEN
        return result

    def fit(self, X, y=None):
        values = self._encode(X, fitting=True)
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(getattr(X,"columns",[f"x{i}" for i in range(values.shape[1])]),dtype=object)
        self.vocabulary_ = [set(values[:,i])- {self.MISSING,self.UNSEEN} for i in range(values.shape[1])]
        categories = [sorted(v | {self.MISSING,self.UNSEEN}) for v in self.vocabulary_]
        self.encoder_ = OneHotEncoder(categories=categories,handle_unknown="error",sparse_output=False)
        self.encoder_.fit(values)
        return self

    def transform(self, X):
        check_is_fitted(self,"encoder_")
        if hasattr(X,"columns") and list(X.columns)!=list(self.feature_names_in_):
            raise ValueError("Categorical input schema changed")
        return self.encoder_.transform(self._encode(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self,"encoder_")
        return self.encoder_.get_feature_names_out(self.feature_names_in_ if input_features is None else input_features)
