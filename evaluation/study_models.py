"""The models compared in this study.

Every model except the area median is trained on log(1 + price) for the
nightly rate, then turned back into dollars. The nights model uses the
target chosen on the training folds. Predictions of nights are clipped to
the same 0–255.5 range as the label. Price predictions are clipped at zero
only, so an expensive guess stays visible.

CatBoost is imported inside the function that builds it. The app does not
import it unless a saved model actually contains it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from evaluation.metrics import regression_metrics
from evaluation.study_protocol import (
    BASELINE_MIN_COUNT,
    LEARNING_RATE,
    MODEL_RANDOM_STATE,
    N_ESTIMATORS,
)


class AreaMedian:
    """Typical rate for the neighbourhood and room type, with a fixed fallback.

    Order: neighbourhood and room type, then the room type in the whole city,
    then the neighbourhood, then the citywide median. A group is used only
    when it has at least BASELINE_MIN_COUNT training listings.
    """

    def __init__(self, min_count: int = BASELINE_MIN_COUNT):
        self.min_count = int(min_count)
        self.global_ = 0.0
        self.pair_: dict[tuple[str, str], float] = {}
        self.room_: dict[str, float] = {}
        self.neighbourhood_: dict[str, float] = {}

    def fit(self, neighbourhood, room, y) -> AreaMedian:
        frame = pd.DataFrame(
            {
                "n": pd.Series(neighbourhood).astype(str).to_numpy(),
                "r": pd.Series(room).astype(str).to_numpy(),
                "y": np.asarray(y, dtype=float),
            }
        )
        self.global_ = float(np.median(frame["y"])) if len(frame) else 0.0
        self.room_ = _median_map(frame, ["r"], self.min_count)
        self.neighbourhood_ = _median_map(frame, ["n"], self.min_count)
        self.pair_ = _median_map(frame, ["n", "r"], self.min_count)
        return self

    def predict(self, neighbourhood, room) -> np.ndarray:
        neigh = pd.Series(neighbourhood).astype(str).to_numpy()
        rooms = pd.Series(room).astype(str).to_numpy()
        out = np.empty(len(neigh), dtype=float)
        for index, (name, kind) in enumerate(zip(neigh, rooms, strict=True)):
            if (name, kind) in self.pair_:
                out[index] = self.pair_[(name, kind)]
            elif kind in self.room_:
                out[index] = self.room_[kind]
            elif name in self.neighbourhood_:
                out[index] = self.neighbourhood_[name]
            else:
                out[index] = self.global_
        return out


def _median_map(frame: pd.DataFrame, keys: list[str], min_count: int) -> dict:
    stats = frame.groupby(keys, sort=False)["y"].agg(["median", "count"])
    kept = stats.loc[stats["count"] >= min_count, "median"]
    if len(keys) == 1:
        return {str(key): float(value) for key, value in kept.items()}
    return {tuple(str(part) for part in key): float(value) for key, value in kept.items()}


class LogModel:
    """A sklearn pipeline whose predict() returns dollars (or nights)."""

    def __init__(
        self, estimator, columns: list[str], *, kind: str = "log", high: float | None = None
    ):
        self.estimator = estimator
        self.columns = list(columns)
        self.kind = kind
        self.high = high
        self.medians_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, y) -> LogModel:
        matrix = frame.loc[:, self.columns]
        stats = np.nanmedian(matrix.to_numpy(dtype=float), axis=0)
        stats = np.where(np.isfinite(stats), stats, 0.0)
        self.medians_ = {
            column: float(value) for column, value in zip(self.columns, stats, strict=True)
        }
        target = np.asarray(y, dtype=float)
        if self.kind == "log":
            target = np.log1p(target)
        self.estimator.fit(matrix, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.estimator.predict(frame.loc[:, self.columns]), dtype=float)
        pred = np.expm1(raw) if self.kind == "log" else raw
        pred = np.maximum(pred, 0.0)
        if self.high is not None:
            pred = np.minimum(pred, float(self.high))
        return pred


class BlendModel:
    """Equal average of two dollar predictions. The weights are not fit."""

    def __init__(self, first: LogModel, second: LogModel):
        self.first = first
        self.second = second
        self.columns = list(first.columns)
        self.medians_ = dict(first.medians_)
        self.kind = first.kind
        self.high = first.high

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return 0.5 * (self.first.predict(frame) + self.second.predict(frame))


def _tree_pipe(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _linear_pipe(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _xgb(depth: int, min_child: float) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=int(depth),
        learning_rate=LEARNING_RATE,
        min_child_weight=float(min_child),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=MODEL_RANDOM_STATE,
        n_jobs=4,
        verbosity=0,
    )


def make_xgb(
    columns: list[str], depth: int, min_child: float, *, kind: str, high: float | None
) -> LogModel:
    return LogModel(_tree_pipe(_xgb(depth, min_child)), columns, kind=kind, high=high)


def make_linear(columns: list[str], *, kind: str, high: float | None) -> LogModel:
    return LogModel(_linear_pipe(LinearRegression()), columns, kind=kind, high=high)


def make_ridge(columns: list[str], alpha: float, *, kind: str, high: float | None) -> LogModel:
    return LogModel(_linear_pipe(Ridge(alpha=float(alpha))), columns, kind=kind, high=high)


def make_forest(
    columns: list[str], depth: int, leaf: int, *, kind: str, high: float | None
) -> LogModel:
    forest = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=int(depth),
        min_samples_leaf=int(leaf),
        max_features=1.0,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=4,
    )
    return LogModel(_tree_pipe(forest), columns, kind=kind, high=high)


def make_network(columns: list[str], *, kind: str, high: float | None) -> LogModel:
    network = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation="relu",
        alpha=1e-3,
        max_iter=300,
        random_state=MODEL_RANDOM_STATE,
        early_stopping=False,
    )
    return LogModel(_linear_pipe(network), columns, kind=kind, high=high)


def make_catboost(columns: list[str], depth: int, *, kind: str, high: float | None) -> LogModel:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=N_ESTIMATORS,
        depth=int(depth),
        learning_rate=LEARNING_RATE,
        l2_leaf_reg=1.0,
        random_seed=MODEL_RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        loss_function="RMSE",
        thread_count=4,
    )
    return LogModel(_tree_pipe(model), columns, kind=kind, high=high)


def cv_mae(
    builder, frame: pd.DataFrame, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> float:
    scores = []
    for train_idx, valid_idx in folds:
        model = builder()
        model.fit(frame.iloc[train_idx], y[train_idx])
        pred = model.predict(frame.iloc[valid_idx])
        scores.append(float(np.mean(np.abs(y[valid_idx] - pred))))
    return float(np.mean(scores))


def cv_area(
    frame: pd.DataFrame, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> float:
    scores = []
    for train_idx, valid_idx in folds:
        model = AreaMedian().fit(
            frame.iloc[train_idx]["neighbourhood_cleansed"],
            frame.iloc[train_idx]["room_type"],
            y[train_idx],
        )
        pred = model.predict(
            frame.iloc[valid_idx]["neighbourhood_cleansed"],
            frame.iloc[valid_idx]["room_type"],
        )
        scores.append(float(np.mean(np.abs(y[valid_idx] - pred))))
    return float(np.mean(scores))


def fit_interval(y: np.ndarray, pred: np.ndarray, min_bin: int) -> dict:
    """10th and 90th residual in each quintile of the out-of-fold prediction."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residual = y - pred
    edges = [float(value) for value in np.quantile(pred, [0.2, 0.4, 0.6, 0.8])]
    bins = np.digitize(pred, edges, right=False)
    global_low = float(np.quantile(residual, 0.10))
    global_high = float(np.quantile(residual, 0.90))
    low, high, counts = [], [], []
    for bin_id in range(5):
        part = residual[bins == bin_id]
        counts.append(int(len(part)))
        if len(part) < min_bin:
            low.append(global_low)
            high.append(global_high)
        else:
            low.append(float(np.quantile(part, 0.10)))
            high.append(float(np.quantile(part, 0.90)))
    return {
        "edges": edges,
        "q10": low,
        "q90": high,
        "bin_counts": counts,
        "global_q10": global_low,
        "global_q90": global_high,
        "method": "oof_price_quintile",
    }


def apply_interval(pred: np.ndarray, table: dict) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    bins = np.digitize(pred, table["edges"], right=False)
    low = pred + np.asarray(table["q10"], dtype=float)[bins]
    high = pred + np.asarray(table["q90"], dtype=float)[bins]
    low = np.maximum(low, 0.0)
    high = np.maximum(high, low)
    return low, high


def oof_predict(builder, frame, y, folds) -> np.ndarray:
    """One prediction per validation row. Rows that are not in a fold stay NaN."""
    pred = np.full(len(frame), np.nan)
    for train_idx, valid_idx in folds:
        model = builder()
        model.fit(frame.iloc[train_idx], y[train_idx])
        pred[valid_idx] = model.predict(frame.iloc[valid_idx])
    return pred


def bootstrap_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    cells: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Percentile intervals from resampling holdout cells, not rows."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    cells = np.asarray(cells)
    point = regression_metrics(y, pred)
    base = regression_metrics(y, baseline)
    removed = _removed(point["mae"], base["mae"])
    groups: dict = {}
    for index, cell in enumerate(cells):
        groups.setdefault(cell, []).append(index)
    keys = list(groups)
    index_groups = [np.asarray(groups[key], dtype=int) for key in keys]
    rng = np.random.default_rng(int(seed))
    samples = {name: [] for name in ("r2", "mae", "median_ape", "baseline_mae", "error_removed")}
    n_keys = len(keys)
    for _ in range(int(n_boot)):
        draw = rng.integers(0, n_keys, size=n_keys)
        idx = np.concatenate([index_groups[int(item)] for item in draw])
        metrics = regression_metrics(y[idx], pred[idx])
        base_metrics = regression_metrics(y[idx], baseline[idx])
        samples["r2"].append(metrics["r2"])
        samples["mae"].append(metrics["mae"])
        samples["median_ape"].append(metrics["median_ape"])
        samples["baseline_mae"].append(base_metrics["mae"])
        samples["error_removed"].append(_removed(metrics["mae"], base_metrics["mae"]))
    packed = {
        "r2": _pack(point["r2"], samples["r2"]),
        "mae": _pack(point["mae"], samples["mae"]),
        "median_ape": _pack(point["median_ape"], samples["median_ape"]),
        "baseline_mae": _pack(base["mae"], samples["baseline_mae"]),
        "error_removed": _pack(removed, samples["error_removed"]),
    }
    return packed


def _removed(model_mae: float, baseline_mae: float) -> float:
    if baseline_mae <= 0:
        return float("nan")
    return float((baseline_mae - model_mae) / baseline_mae)


def _pack(point: float, samples: list[float]) -> dict:
    low, high = np.nanpercentile(np.asarray(samples, dtype=float), [2.5, 97.5])
    return {"point": float(point), "low": float(low), "high": float(high)}


def dollar_effects(model, frame: pd.DataFrame, idx: np.ndarray, limit: int = 200) -> list[dict]:
    """Mean absolute dollar change from setting one feature to its training median."""
    rng = np.random.default_rng(MODEL_RANDOM_STATE)
    take = idx if len(idx) <= limit else rng.choice(idx, size=limit, replace=False)
    base_rows = frame.iloc[np.asarray(take)].copy()
    base = model.predict(base_rows)
    rows = []
    medians = model.medians_
    for column in model.columns:
        altered = base_rows.copy()
        altered[column] = medians[column]
        delta = base - model.predict(altered)
        rows.append(
            {
                "feature": column,
                "mean_abs_dollars": float(np.mean(np.abs(delta))),
                "mean_dollars": float(np.mean(delta)),
            }
        )
    rows.sort(key=lambda item: item["mean_abs_dollars"], reverse=True)
    return rows


def shap_means(
    model: LogModel, frame: pd.DataFrame, idx: np.ndarray, limit: int = 400
) -> list[dict] | None:
    """Mean absolute TreeSHAP on a sample. Only XGBoost has this."""
    estimator = model.estimator.named_steps.get("model")
    if not isinstance(estimator, XGBRegressor):
        return None
    rng = np.random.default_rng(MODEL_RANDOM_STATE)
    take = idx if len(idx) <= limit else rng.choice(idx, size=limit, replace=False)
    matrix = frame.iloc[np.asarray(take)].loc[:, model.columns]
    imputed = model.estimator.named_steps["imputer"].transform(matrix)
    import xgboost as xgb

    contrib = estimator.get_booster().predict(
        xgb.DMatrix(imputed, feature_names=model.columns), pred_contribs=True
    )
    values = np.asarray(contrib, dtype=float)[:, :-1]
    rows = []
    for index, column in enumerate(model.columns):
        rows.append(
            {
                "feature": column,
                "mean_abs_shap": float(np.mean(np.abs(values[:, index]))),
                "mean_shap": float(np.mean(values[:, index])),
            }
        )
    rows.sort(key=lambda item: item["mean_abs_shap"], reverse=True)
    return rows


def direction_checks(model, frame: pd.DataFrame, train_idx: np.ndarray) -> list[dict]:
    """Move one feature from a low value to a high value, holding the rest typical."""
    medians = model.medians_
    base = pd.DataFrame([{column: medians[column] for column in model.columns}])
    train = frame.iloc[train_idx]
    rows = []
    for column in model.columns:
        is_flag = (
            column.startswith("is_")
            or column.startswith("has_")
            or column
            in {
                "bath_is_shared",
                "is_superhost",
            }
        )
        if is_flag:
            low, high = 0.0, 1.0
        else:
            column_values = train[column].to_numpy(dtype=float)
            if np.isfinite(column_values).sum() < 20:
                continue
            low = float(np.nanpercentile(column_values, 25))
            high = float(np.nanpercentile(column_values, 75))
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                continue
        low_row = base.copy()
        high_row = base.copy()
        low_row[column] = low
        high_row[column] = high
        delta = float(model.predict(high_row)[0] - model.predict(low_row)[0])
        rows.append({"feature": column, "low": low, "high": high, "delta": delta})
    return rows
