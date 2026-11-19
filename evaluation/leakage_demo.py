"""Reproduce the old area-median target on Austin, as a demonstration.

This is not a result. It fits on the whole Austin file, on purpose, to show
why the previous R² near 0.97 was not a measure of listing-level accuracy.

Two separate problems get mixed together there:

1. The training target was a neighbourhood median instead of the listing's
   own price. A smooth target is easy to predict, and that R² does not
   transfer to the raw price.
2. Those medians were computed before the split, so a listing helped
   define its own target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from evaluation.metrics import regression_metrics


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def austin_leakage_demo(frame: pd.DataFrame) -> dict:
    y = frame["price_clean"].to_numpy(dtype=float)
    bedrooms = np.clip(np.rint(frame["bedrooms"].to_numpy(dtype=float)), 0, 6).astype(int)
    coords = frame[["latitude", "longitude"]].to_numpy(dtype=float)
    n_clusters = int(max(8, min(15, len(frame) // 400)))
    clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(coords)

    table = pd.DataFrame({"cluster": clusters, "bedrooms": bedrooms, "y": y})
    grouped = table.groupby(["cluster", "bedrooms"])["y"]
    leaky_median = grouped.transform("median").to_numpy(dtype=float)
    counts = grouped.transform("count").to_numpy(dtype=float)
    sums = grouped.transform("sum").to_numpy(dtype=float)
    loo = np.full(len(y), np.nan)
    multi = counts > 1
    loo[multi] = (sums[multi] - y[multi]) / (counts[multi] - 1)

    features = frame[["latitude", "longitude", "bedrooms", "bathrooms"]].apply(
        pd.to_numeric, errors="coerce"
    )
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in KFold(5, shuffle=True, random_state=42).split(features):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(features.iloc[train_idx], leaky_median[train_idx])
        oof[test_idx] = model.predict(features.iloc[test_idx])

    versus_target = regression_metrics(leaky_median, oof)
    versus_price = regression_metrics(y, oof)
    return {
        "role": "demonstration_of_the_old_target_not_a_reported_result",
        "n": int(len(y)),
        "n_clusters": n_clusters,
        "correlation_price_with_median_that_includes_itself": _corr(y, leaky_median),
        "correlation_price_with_leave_one_out_median": _corr(y[multi], loo[multi]),
        "cv_r2_against_smoothed_target": versus_target["r2"],
        "cv_mae_against_smoothed_target": versus_target["mae"],
        "cv_r2_against_actual_price": versus_price["r2"],
        "cv_mae_against_actual_price": versus_price["mae"],
        "cv_rmse_against_actual_price": versus_price["rmse"],
    }
