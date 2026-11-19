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
from features.columns import KNN_COLUMNS, MONOTONE_DOWN, MONOTONE_UP, SMOOTH_COLUMNS
from features.fold import FoldEncoder, KernelSmoother
from features.text import FoldText
from lightgbm import early_stopping
from models.regressors import FIXED_PARAMS, make_catboost, make_lightgbm, make_xgboost
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.protocol import (
    EARLY_STOPPING_ROUNDS,
    KERNEL_BANDWIDTH_KM,
    KERNEL_NEIGHBORS,
    KNN_K,
    MAX_TREES,
    N_TEXT_COMPONENTS,
    TARGET_PRIOR,
    TEXT_MAX_FEATURES,
    TEXT_MIN_DF,
)
from evaluation.splits import holdout_groups

SPATIAL_COLUMNS = SMOOTH_COLUMNS + KNN_COLUMNS


class DesignState:
    """Fold-fit pieces that have to travel with the matrix.

    `fold` is the neighbourhood / nearest-listing encoder. `kernel` is the
    distance-weighted price smoother. `text` is the TF-IDF compression.
    Any of them can be missing. The trees never see this object; they see
    the columns it already wrote into the frame.
    """

    def __init__(self) -> None:
        self.fold: FoldEncoder | None = None
        self.kernel: KernelSmoother | None = None
        self.text: FoldText | None = None


def params_for_depth(depth: int) -> dict:
    params = dict(FIXED_PARAMS)
    params["max_depth"] = int(depth)
    return params


def _as_frame(X: pd.DataFrame) -> pd.DataFrame:
    return X.reset_index(drop=True)


def monotone_spec(columns: list[str], library: str):
    """Constraint aligned to `columns`. 1 means the prediction cannot fall as the column rises."""
    spec = []
    for column in columns:
        if column in MONOTONE_UP:
            spec.append(1)
        elif column in MONOTONE_DOWN:
            spec.append(-1)
        else:
            spec.append(0)
    if library == "lightgbm":
        return spec
    return {
        column: value
        for column, value in zip(columns, spec, strict=True)
        if value or library == "xgboost"
    }


def design_matrices(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    columns: list[str],
    spatial_mode: str | bool,
    text_mode: str = "none",
    text_encoder: FoldText | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, DesignState]:
    """Train/eval matrices. Price encodings and text SVD are fit on train_idx only.

    `spatial_mode` is "none", "smooth", "knn", or "kernel". True is accepted
    as "smooth" plus knn so an older caller still builds every neighbourhood
    column. `text_mode` "svd" fits a TF-IDF compression on the training
    listings' own ad copy. Pass `text_encoder` to reuse one already fit on
    a wider training set (the pooled model shares a vocabulary across cities).
    """
    mode = _spatial_mode(spatial_mode)
    if text_mode not in {"none", "svd"}:
        raise ValueError(f"Unknown text mode: {text_mode}")
    y = frame["price_clean"].to_numpy(dtype=float)
    log_y = np.log1p(y[train_idx])
    X_train = _as_frame(frame.iloc[train_idx][list(columns)].copy())
    X_eval = _as_frame(frame.iloc[eval_idx][list(columns)].copy())
    state = DesignState()
    if mode in {"smooth", "knn", "both"}:
        state.fold = FoldEncoder(k=KNN_K, prior=TARGET_PRIOR)
        state.fold.fit(
            frame["latitude"].to_numpy()[train_idx],
            frame["longitude"].to_numpy()[train_idx],
            frame["neighbourhood_cleansed"].to_numpy()[train_idx],
            log_y,
            frame["bedrooms"].to_numpy()[train_idx],
        )
        train_spatial = state.fold.transform_train(
            frame["latitude"].to_numpy()[train_idx],
            frame["longitude"].to_numpy()[train_idx],
            frame["neighbourhood_cleansed"].to_numpy()[train_idx],
            frame["bedrooms"].to_numpy()[train_idx],
        )
        eval_spatial = state.fold.transform_new(
            frame["latitude"].to_numpy()[eval_idx],
            frame["longitude"].to_numpy()[eval_idx],
            frame["neighbourhood_cleansed"].to_numpy()[eval_idx],
            frame["bedrooms"].to_numpy()[eval_idx],
        )
        names = _spatial_names(mode)
        for name, train_values, eval_values in zip(
            names,
            _select_spatial(train_spatial, mode),
            _select_spatial(eval_spatial, mode),
            strict=True,
        ):
            X_train[name] = train_values
            X_eval[name] = eval_values
    if mode == "kernel":
        state.kernel = KernelSmoother(
            bandwidth_km=KERNEL_BANDWIDTH_KM, n_neighbors=KERNEL_NEIGHBORS
        )
        state.kernel.fit(
            frame["latitude"].to_numpy()[train_idx],
            frame["longitude"].to_numpy()[train_idx],
            log_y,
        )
        X_train["kernel_log_price"] = state.kernel.transform(
            frame["latitude"].to_numpy()[train_idx],
            frame["longitude"].to_numpy()[train_idx],
            row_index=np.arange(len(train_idx)),
        )
        X_eval["kernel_log_price"] = state.kernel.transform(
            frame["latitude"].to_numpy()[eval_idx],
            frame["longitude"].to_numpy()[eval_idx],
            row_index=None,
        )
    if text_mode == "svd":
        state.text = text_encoder or FoldText(
            n_components=N_TEXT_COMPONENTS,
            max_features=TEXT_MAX_FEATURES,
            min_df=TEXT_MIN_DF,
        )
        train_text = frame["listing_text"].to_numpy()[train_idx]
        if text_encoder is None:
            state.text.fit(train_text)
        train_reduced = state.text.transform(train_text)
        eval_reduced = state.text.transform(frame["listing_text"].to_numpy()[eval_idx])
        for index, name in enumerate(state.text.column_names):
            X_train[name] = train_reduced[:, index]
            X_eval[name] = eval_reduced[:, index]
    # Trees want numbers. A stray object column would be read as a category.
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_eval = X_eval.apply(pd.to_numeric, errors="coerce")
    return X_train, log_y, X_eval, y[eval_idx], state


def _spatial_mode(spatial_mode: str | bool) -> str:
    if spatial_mode is True:
        return "both"
    if spatial_mode is False or spatial_mode is None:
        return "none"
    if spatial_mode not in {"none", "smooth", "knn", "both", "kernel"}:
        raise ValueError(f"Unknown spatial mode: {spatial_mode}")
    return spatial_mode


def _spatial_names(mode: str) -> list[str]:
    if mode == "smooth":
        return list(SMOOTH_COLUMNS)
    if mode == "knn":
        return list(KNN_COLUMNS)
    if mode == "both":
        return list(SPATIAL_COLUMNS)
    return []


def _select_spatial(values: tuple[np.ndarray, np.ndarray, np.ndarray], mode: str):
    neigh, bed, knn = values
    if mode == "smooth":
        return (neigh, bed)
    if mode == "knn":
        return (knn,)
    return values


def _predict_log(model, X: pd.DataFrame, library: str) -> np.ndarray:
    trees = getattr(model, "_capscout_trees", None)
    if library == "xgboost" and trees:
        logged = model.predict(X, iteration_range=(0, int(trees)))
    elif library == "lightgbm" and trees:
        logged = model.predict(X, num_iteration=int(trees))
    else:
        logged = model.predict(X)
    return np.asarray(logged, dtype=float).reshape(-1)


def _predict_dollars(model, X: pd.DataFrame, library: str) -> np.ndarray:
    return np.maximum(np.expm1(_predict_log(model, X, library)), 0.0)


def _new_model(library: str, params: dict, n_estimators: int, *, monotone, early_stopping: bool):
    constraint = monotone_spec(list(params.get("_columns") or []), library) if monotone else None
    # The column list rides along on params so the constraint lines up with
    # the matrix. It is not a hyperparameter and the constructors ignore it.
    if library == "xgboost":
        return make_xgboost(
            params,
            n_estimators,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS if early_stopping else None,
            monotone=constraint,
        )
    if library == "lightgbm":
        return make_lightgbm(params, n_estimators, monotone=constraint)
    if library == "catboost":
        return make_catboost(params, n_estimators, monotone=constraint or None)
    raise ValueError(f"Unknown library: {library}")


def _weight_slice(sample_weight: np.ndarray | None, index: np.ndarray) -> np.ndarray | None:
    if sample_weight is None:
        return None
    return np.asarray(sample_weight, dtype=float)[index]


def _fit_model(model, library: str, X, y, sample_weight, eval_X=None, eval_y=None) -> None:
    """Train one library. `sample_weight` is omitted when it is None.

    Passing None is not the same as omitting it for every constructor, so
    the argument is only added when a tail weight was actually built.
    """
    kwargs: dict = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)
    if library == "xgboost":
        if eval_X is not None:
            kwargs["eval_set"] = [(eval_X, eval_y)]
            kwargs["verbose"] = False
        model.fit(X, y, **kwargs)
    elif library == "lightgbm":
        if eval_X is not None:
            kwargs["eval_set"] = [(eval_X, eval_y)]
            kwargs["callbacks"] = [early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
        model.fit(X, y, **kwargs)
    else:
        if eval_X is not None:
            kwargs["eval_set"] = (eval_X, eval_y)
            kwargs["use_best_model"] = True
            kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        model.fit(X, y, **kwargs)


def fit_boosted(
    library: str,
    params: dict,
    X: pd.DataFrame,
    y_log: np.ndarray,
    groups: np.ndarray,
    *,
    monotone: bool = False,
    sample_weight: np.ndarray | None = None,
):
    """Early-stopped model. `_capscout_trees` is the tree count to use.

    The stopping slice is a spatial cut of the training rows. It is not the
    validation fold and not the locked test. During comparison we predict
    with that early-stopped fit, so every candidate leaves the same rows out.
    `sample_weight` is the tail weight. It is None for every model that is
    not the tail-weighted candidate, so those fits stay comparable to v3.
    """
    X = _as_frame(X)
    y_log = np.asarray(y_log, dtype=float)
    groups = np.asarray(groups)
    fitted = dict(params)
    fitted["_columns"] = list(X.columns)
    n_groups = len(np.unique(groups))
    if n_groups >= 8 and len(X) >= 200:
        inner_train, inner_valid = holdout_groups(groups, 0.2, seed=0)
        model = _new_model(library, fitted, MAX_TREES, monotone=monotone, early_stopping=True)
        _fit_model(
            model,
            library,
            X.iloc[inner_train],
            y_log[inner_train],
            _weight_slice(sample_weight, inner_train),
            X.iloc[inner_valid],
            y_log[inner_valid],
        )
        if library == "xgboost":
            model._capscout_trees = int(model.best_iteration) + 1
        elif library == "lightgbm":
            model._capscout_trees = int(model.best_iteration_)
        else:
            model._capscout_trees = int(model.tree_count_)
        return model

    model = _new_model(
        library, fitted, min(MAX_TREES, 200), monotone=monotone, early_stopping=False
    )
    _fit_model(model, library, X, y_log, sample_weight)
    model._capscout_trees = None
    return model


def refit_boosted(
    library: str,
    params: dict,
    n_trees: int,
    X: pd.DataFrame,
    y_log: np.ndarray,
    *,
    monotone: bool = False,
    sample_weight: np.ndarray | None = None,
):
    """Refit on every development row with the tree count early stopping chose."""
    n_trees = max(int(n_trees), 20)
    X = _as_frame(X)
    fitted = dict(params)
    fitted["_columns"] = list(X.columns)
    model = _new_model(library, fitted, n_trees, monotone=monotone, early_stopping=False)
    _fit_model(model, library, X, np.asarray(y_log, dtype=float), sample_weight)
    model._capscout_trees = None
    return model


def fit_quantile(
    library: str,
    params: dict,
    n_trees: int,
    X: pd.DataFrame,
    y_log: np.ndarray,
    alpha: float,
):
    """Pinball loss at `alpha`. Unconstrained: LightGBM rejects monotone + quantile.

    The tree count is taken from the point model. It is not re-searched,
    because a search here would be a second tuning loop on the same folds.
    """
    n_trees = max(int(n_trees), 20)
    X = _as_frame(X)
    y_log = np.asarray(y_log, dtype=float)
    if library == "xgboost":
        model = make_xgboost(
            params,
            n_trees,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
        )
    elif library == "lightgbm":
        model = make_lightgbm(params, n_trees, objective="quantile", alpha=alpha)
    elif library == "catboost":
        model = make_catboost(params, n_trees, loss=f"Quantile:alpha={alpha}")
    else:
        raise ValueError(library)
    model.fit(X, y_log)
    model._capscout_trees = None
    return model


def predict_dollars(model, X: pd.DataFrame, library: str) -> np.ndarray:
    return _predict_dollars(model, _as_frame(X), library)


def predict_log(model, X: pd.DataFrame, library: str) -> np.ndarray:
    return _predict_log(model, _as_frame(X), library)


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
