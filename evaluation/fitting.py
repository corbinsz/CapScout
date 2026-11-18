"""Fit the models that the protocol compares.

Boosted trees are trained on log1p(price). Early stopping picks the number
of trees from a spatial slice of the training rows; that slice is not the
validation fold and not the locked test. During model comparison we predict
with the early-stopped fit directly (it has not been refit on the stopping
slice), so every candidate is treated the same. The one model that scores
the locked test is refit on all development rows with that tree count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from features.fold import FoldEncoder
from lightgbm import early_stopping
from models.regressors import FIXED_PARAMS, make_lightgbm, make_xgboost
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.protocol import EARLY_STOPPING_ROUNDS, KNN_K, MAX_TREES, TARGET_PRIOR
from evaluation.splits import holdout_groups

SPATIAL_COLUMNS = ["neigh_log_price", "neigh_bed_log_price", "knn_log_price"]


def params_for_depth(depth: int) -> dict:
    params = dict(FIXED_PARAMS)
    params["max_depth"] = int(depth)
    return params


def _as_frame(X: pd.DataFrame) -> pd.DataFrame:
    return X.reset_index(drop=True)


def design_matrices(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    columns: list[str],
    use_spatial: bool,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, FoldEncoder | None]:
    """Train/eval matrices. Spatial columns are fit on train_idx only."""
    y = frame["price_clean"].to_numpy(dtype=float)
    log_y = np.log1p(y[train_idx])
    X_train = _as_frame(frame.iloc[train_idx][list(columns)].copy())
    X_eval = _as_frame(frame.iloc[eval_idx][list(columns)].copy())
    encoder = None
    if use_spatial:
        encoder = FoldEncoder(k=KNN_K, prior=TARGET_PRIOR)
        encoder.fit(
            frame["latitude"].to_numpy()[train_idx],
            frame["longitude"].to_numpy()[train_idx],
            frame["neighbourhood_cleansed"].to_numpy()[train_idx],
            log_y,
            frame["bedrooms"].to_numpy()[train_idx],
        )
        train_spatial = encoder.transform_train(
            frame["latitude"].to_numpy()[train_idx],
            frame["longitude"].to_numpy()[train_idx],
            frame["neighbourhood_cleansed"].to_numpy()[train_idx],
            frame["bedrooms"].to_numpy()[train_idx],
        )
        eval_spatial = encoder.transform_new(
            frame["latitude"].to_numpy()[eval_idx],
            frame["longitude"].to_numpy()[eval_idx],
            frame["neighbourhood_cleansed"].to_numpy()[eval_idx],
            frame["bedrooms"].to_numpy()[eval_idx],
        )
        for name, train_values, eval_values in zip(
            SPATIAL_COLUMNS, train_spatial, eval_spatial, strict=True
        ):
            X_train[name] = train_values
            X_eval[name] = eval_values
    return X_train, log_y, X_eval, y[eval_idx], encoder


def _predict_dollars(model, X: pd.DataFrame, library: str) -> np.ndarray:
    trees = getattr(model, "_capscout_trees", None)
    if library == "xgboost" and trees:
        logged = model.predict(X, iteration_range=(0, int(trees)))
    elif library == "lightgbm" and trees:
        logged = model.predict(X, num_iteration=int(trees))
    else:
        logged = model.predict(X)
    return np.maximum(np.expm1(np.asarray(logged, dtype=float)), 0.0)


def fit_boosted(library: str, params: dict, X: pd.DataFrame, y_log: np.ndarray, groups: np.ndarray):
    """Early-stopped model. `_capscout_trees` is the tree count to use."""
    X = _as_frame(X)
    y_log = np.asarray(y_log, dtype=float)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups >= 8 and len(X) >= 200:
        inner_train, inner_valid = holdout_groups(groups, 0.2, seed=0)
        if library == "xgboost":
            model = make_xgboost(params, MAX_TREES, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.fit(
                X.iloc[inner_train],
                y_log[inner_train],
                eval_set=[(X.iloc[inner_valid], y_log[inner_valid])],
                verbose=False,
            )
            model._capscout_trees = int(model.best_iteration) + 1
        else:
            model = make_lightgbm(params, MAX_TREES)
            model.fit(
                X.iloc[inner_train],
                y_log[inner_train],
                eval_set=[(X.iloc[inner_valid], y_log[inner_valid])],
                callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            model._capscout_trees = int(model.best_iteration_)
        return model

    if library == "xgboost":
        model = make_xgboost(params, min(MAX_TREES, 200), early_stopping_rounds=None)
    else:
        model = make_lightgbm(params, min(MAX_TREES, 200))
    model.fit(X, y_log)
    model._capscout_trees = None
    return model


def refit_boosted(library: str, params: dict, n_trees: int, X: pd.DataFrame, y_log: np.ndarray):
    """Refit on every development row with the tree count early stopping chose."""
    n_trees = max(int(n_trees), 20)
    X = _as_frame(X)
    if library == "xgboost":
        model = make_xgboost(params, n_trees, early_stopping_rounds=None)
    else:
        model = make_lightgbm(params, n_trees)
    model.fit(X, np.asarray(y_log, dtype=float))
    model._capscout_trees = None
    return model


def predict_dollars(model, X: pd.DataFrame, library: str) -> np.ndarray:
    return _predict_dollars(model, _as_frame(X), library)


def fit_predict_ridge(X_train: pd.DataFrame, y_log: np.ndarray, X_eval: pd.DataFrame) -> np.ndarray:
    """Ridge on the log price. Medians and scaling are fit on the train fold only."""
    medians = X_train.median(numeric_only=True)
    train = X_train.fillna(medians).fillna(0.0)
    eval_ = X_eval.fillna(medians).fillna(0.0)
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-2, 3, 12))),
        ]
    )
    pipe.fit(train, np.asarray(y_log, dtype=float))
    return np.maximum(np.expm1(pipe.predict(eval_)), 0.0)


def _keys(neighbourhood: np.ndarray, room: np.ndarray, bedrooms: np.ndarray | None) -> np.ndarray:
    names = np.asarray(neighbourhood).astype(str)
    rooms = np.asarray(room).astype(str)
    if bedrooms is None:
        return np.array([f"{n}|{r}" for n, r in zip(names, rooms, strict=True)])
    bins = np.clip(np.rint(np.asarray(bedrooms, dtype=float)), 0, 6).astype(int)
    return np.array([f"{n}|{r}|{b}" for n, r, b in zip(names, rooms, bins, strict=True)])


def _lookup(keys: np.ndarray, mapping: dict[str, float], default: np.ndarray) -> np.ndarray:
    return np.array([mapping.get(key, default[i]) for i, key in enumerate(keys)], dtype=float)


def median_baseline(
    neighbourhood: np.ndarray,
    room: np.ndarray,
    bedrooms: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    by_bedroom: bool,
) -> np.ndarray:
    """Training-fold median.

    Primary key is neighbourhood × room type, or that pair plus a bedroom bin.
    Fallback is neighbourhood, then the training median. The evaluation
    listing's own price is not in the map.
    """
    y_train = y[train_idx]
    global_median = float(np.median(y_train))
    neigh_keys = np.asarray(neighbourhood).astype(str)
    neigh_map = (
        pd.Series(y_train)
        .groupby(pd.Series(neigh_keys[train_idx]), sort=False)
        .median()
        .astype(float)
        .to_dict()
    )
    neigh_default = _lookup(neigh_keys[eval_idx], neigh_map, np.full(len(eval_idx), global_median))

    primary = _keys(neighbourhood, room, bedrooms if by_bedroom else None)
    primary_map = (
        pd.Series(y_train)
        .groupby(pd.Series(primary[train_idx]), sort=False)
        .median()
        .astype(float)
        .to_dict()
    )
    if not by_bedroom:
        return _lookup(primary[eval_idx], primary_map, neigh_default)

    room_keys = _keys(neighbourhood, room, None)
    room_map = (
        pd.Series(y_train)
        .groupby(pd.Series(room_keys[train_idx]), sort=False)
        .median()
        .astype(float)
        .to_dict()
    )
    room_default = _lookup(room_keys[eval_idx], room_map, neigh_default)
    return _lookup(primary[eval_idx], primary_map, room_default)
