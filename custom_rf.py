# custom_rf.py
# -*- coding: utf-8 -*-

"""Custom RandomForest wrapper with Importance weighting based on tillage class-imbalance and crop class (cdl_cropType).

This module provides:
- `calculate_custom_weights`: per-class weights w_c ∝ (1 / n_c) ** a (normalized).
- `CustomWeightedRF`: a scikit-learn compatible classifier that multiplies
   target-based weights with feature-based weights from the column
   `cdl_cropType'.

Notes
-----
- Parameter `a` controls how aggressively minority classes are upweighted:
    a = 0.0  -> uniform weights
    a = 1.0  -> inverse frequency
    a > 1.0  -> stronger emphasis on rarer classes
- Sample weights used for fitting are:  weight(y_i) * weight(X_i['cdl_cropType'])
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "calculate_custom_weights",
    "CustomWeightedRF",
    "alias_for_pickle",
]


def calculate_custom_weights(y_arr: np.ndarray, a: float) -> Dict[Any, float]:
    """
    Compute normalized per-class weights proportional to (1 / n_c) ** a.

    Parameters
    ----------
    y_arr : np.ndarray
        Array of class labels (1-D).
    a : float
        Exponent controlling imbalance correction intensity.

    Returns
    -------
    dict
        Mapping {class_label: weight}, normalized so weights sum to 1 over classes.
    """
    y_arr = np.asarray(y_arr)
    unique_classes, class_counts = np.unique(y_arr, return_counts=True)

    # Avoid divide-by-zero
    class_counts = np.asarray(class_counts, dtype=float)
    class_counts[class_counts == 0] = 1.0

    raw = (1.0 / class_counts) ** float(a)
    denom = raw.sum()
    if denom == 0:
        raw = np.ones_like(raw, dtype=float)
        denom = raw.sum()
    weights = raw / denom
    return {cls: w for cls, w in zip(unique_classes, weights)}

# ----------------------------
# CustomWeightedRF wrapper
# ----------------------------
class CustomWeightedRF(BaseEstimator, ClassifierMixin):
    """
    RandomForest classifier with multiplicative sample weights derived from:
      (1) class-imbalance on the target y, and
      (2) Categorical feature `cdl_cropType` in X.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_depth : int or None, default=None
        Max tree depth.
    min_samples_split : int, default=2
        Min samples required to split an internal node.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    a : float, default=0.0
        Imbalance exponent; 0 => uniform, 1 => inverse-freq, >1 => stronger.
    random_state : int or None, default=None
        Seed for reproducibility.
    **rf_kwargs : Any
        Additional keyword args forwarded to `RandomForestClassifier(...)`.
        Use `set_params(rf__<param>=value)` to set later as well.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        bootstrap: bool = True,
        a: float = 0.0,
        random_state: Optional[int] = None,
        **rf_kwargs: Any,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.a = a
        self.random_state = random_state
        self.rf_kwargs = dict(rf_kwargs)

    # --- sklearn API: parameter handling ---
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return estimator parameters, including RF kwargs under `rf__<name>`.
        """
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "bootstrap": self.bootstrap,
            "a": self.a,
            "random_state": self.random_state,
        }
        for k, v in self.rf_kwargs.items():
            params[f"rf__{k}"] = v
        return params

    def set_params(self, **params: Any):
        """
        Set parameters; keys prefixed by `rf__` are forwarded to the inner RF.
        """
        rf_kwargs = {k[4:]: v for k, v in list(params.items()) if k.startswith("rf__")}
        for k in list(rf_kwargs.keys()):
            params.pop(f"rf__{k}", None)

        for k, v in params.items():
            setattr(self, k, v)

        self.rf_kwargs.update(rf_kwargs)
        return self

    # --- internal helper ---
    def _build_rf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            **self.rf_kwargs,
        )

    # --- sklearn API: fit/predict ---
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_kwargs):
        """
        Fit the underlying RandomForest using multiplicative sample weights.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix. `cdl_cropType` is used to derive
            additional feature-based weights via `calculate_custom_weights`.
        y : pandas.Series
            Target labels.
        **fit_kwargs
            Passed through to `RandomForestClassifier.fit`.

        Returns
        -------
        self : CustomWeightedRF
            Fitted estimator.
        """
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")

        # Target-based weights
        tgt_w = calculate_custom_weights(np.asarray(y), self.a)
        target_weights = np.array([tgt_w[val] for val in y])

        # Feature-based weights from a categorical column
        feat_col = "cdl_cropType"
        if feat_col in X.columns:
            fw = calculate_custom_weights(X[feat_col].values, self.a)
            feature_weights = X[feat_col].map(fw).to_numpy(dtype=float)
        else:
            feature_weights = np.ones(len(X), dtype=float)

        sample_weights = target_weights * feature_weights

        self.rf_ = self._build_rf()
        self.rf_.fit(X, y, sample_weight=sample_weights, **fit_kwargs)
        self.classes_ = self.rf_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the provided data.
        """
        check_is_fitted(self, "rf_")
        return self.rf_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the provided data.
        """
        check_is_fitted(self, "rf_")
        return self.rf_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Return feature importances from the fitted RandomForest.
        """
        check_is_fitted(self, "rf_")
        return self.rf_.feature_importances_

# ----------------------------
# Compatibility helper for OLD pickles
# ----------------------------
def alias_for_pickle() -> None:
    """
    Register `CustomWeightedRF` under `__main__` for unpickling old models that
    were saved from notebooks/scripts where the class lived in `__main__`.
    """
    import sys as _sys

    _sys.modules.setdefault("__main__", _sys.modules["builtins"])
    setattr(_sys.modules["__main__"], "CustomWeightedRF", CustomWeightedRF)
