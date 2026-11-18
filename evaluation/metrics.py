"""Dollar-scale regression metrics. Never computed on a smoothed target."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R², MAE, and RMSE in dollars per night."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # A log model can theoretically predict a hair under zero after expm1
    # only if the logged prediction is negative and large. Floor at zero so
    # MAE stays in dollars a person would quote. Do not clip the top end:
    # that would hide over-prediction.
    y_pred = np.maximum(y_pred, 0.0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "n": int(len(y_true)),
    }


def summarize_folds(folds: list[dict[str, float]]) -> dict[str, float | dict[str, float]]:
    """Mean and sample standard deviation across spatial folds."""
    summary: dict[str, float | dict[str, float]] = {"n": int(sum(f["n"] for f in folds))}
    for key in ("r2", "mae", "rmse"):
        values = np.array([f[key] for f in folds], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[key] = {"mean": float(values.mean()), "std": std}
    return summary
