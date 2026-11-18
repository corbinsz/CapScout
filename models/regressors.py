"""Gradient-boosted trees on log nightly price, and the dollar-scale wrapper.

Nightly rates are right-skewed. Squared error on dollars spends the model's
capacity on a few expensive listings. Squared error on log1p(price) treats
a 20% miss on a $100 night like a 20% miss on a $400 night, which matches
how these prices actually vary. Metrics in the report are always converted
back to dollars with expm1. The wrapper does that conversion in `predict`
so the local app never has to know the target was logged.

Hyperparameters that are not selected on Austin are fixed here on purpose.
A wide search on every city would use the same folds we report.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Frozen before looking at held-out cities. Depth and the library (XGBoost
# vs LightGBM) are the only choices made from Austin spatial CV.
FIXED_PARAMS = {
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_depth": 6,
}

# The previous pipeline's quick-train defaults. Used only for the "before"
# comparison, so the new search cannot retune the old specification.
PREVIOUS_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


class LogPriceModel:
    """Pickleable estimator whose predict() returns dollars per night."""

    def __init__(self, estimator, feature_names: list[str]):
        self.estimator = estimator
        self.feature_names = list(feature_names)

    def predict(self, X) -> np.ndarray:
        logged = np.asarray(self.estimator.predict(X), dtype=float)
        return np.expm1(logged)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator.feature_importances_

    def get_booster(self):
        return self.estimator.get_booster()


def make_xgboost(
    params: dict, n_estimators: int, *, early_stopping_rounds: int | None = None
) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=params["learning_rate"],
        max_depth=int(params["max_depth"]),
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=early_stopping_rounds,
    )


def make_lightgbm(params: dict, n_estimators: int) -> LGBMRegressor:
    depth = int(params["max_depth"])
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=params["learning_rate"],
        max_depth=depth,
        num_leaves=min(2**depth - 1, 63),
        min_child_samples=max(int(params["min_child_weight"]) * 4, 20),
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def make_previous_xgboost() -> XGBRegressor:
    """Dollar-target model with the old defaults. No early stopping."""
    return XGBRegressor(
        **PREVIOUS_PARAMS,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
