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
    params: dict,
    n_estimators: int,
    *,
    early_stopping_rounds: int | None = None,
    monotone: dict | None = None,
    objective: str = "reg:squarederror",
    quantile_alpha: float | None = None,
) -> XGBRegressor:
    kwargs = {}
    if quantile_alpha is not None:
        kwargs["quantile_alpha"] = quantile_alpha
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=params["learning_rate"],
        max_depth=int(params["max_depth"]),
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        objective=objective,
        tree_method="hist",
        random_state=42,
        # One thread. Histogram splits move the third decimal when every
        # core is used, and a second run of this study should match the first.
        n_jobs=1,
        verbosity=0,
        early_stopping_rounds=early_stopping_rounds,
        monotone_constraints=monotone,
        **kwargs,
    )


def make_lightgbm(
    params: dict,
    n_estimators: int,
    *,
    monotone: list[int] | None = None,
    objective: str = "regression",
    alpha: float | None = None,
) -> LGBMRegressor:
    depth = int(params["max_depth"])
    kwargs = {}
    if alpha is not None:
        kwargs["alpha"] = alpha
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=params["learning_rate"],
        max_depth=depth,
        num_leaves=min(2**depth - 1, 63),
        min_child_samples=max(int(params["min_child_weight"]) * 4, 20),
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        objective=objective,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        monotone_constraints=monotone,
        **kwargs,
    )


def make_catboost(
    params: dict,
    n_estimators: int,
    *,
    monotone: dict | None = None,
    loss: str = "RMSE",
):
    """CatBoost on the same depth and learning rate as the other two libraries.

    `allow_writing_files` stays off so a training run does not drop a
    catboost_info directory into the repo. Depth is the same integer the
    protocol searched for XGBoost; it is the analogous knob, not a claim
    that the trees split identically.
    """
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=n_estimators,
        depth=int(params["max_depth"]),
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["reg_lambda"],
        # subsample is ignored unless the bootstrap is Bernoulli. Bayesian
        # bootstrap is the CatBoost default, and it would raise here.
        bootstrap_type="Bernoulli",
        subsample=params["subsample"],
        rsm=params["colsample_bytree"],
        loss_function=loss,
        monotone_constraints=monotone,
        random_seed=42,
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
    )


def make_previous_xgboost() -> XGBRegressor:
    """Dollar-target model with the old defaults. No early stopping."""
    return XGBRegressor(
        **PREVIOUS_PARAMS,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
