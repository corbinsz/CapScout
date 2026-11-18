"""80% prediction intervals from development residuals.

The radius is the split-conformal quantile of absolute residuals in log
price. It is computed on spatial validation folds that the point model was
not trained on. It is not computed on the locked test. Applying one radius
in log space makes the dollar interval wider for expensive listings, which
is what the residuals do.

Exchangeability is not exactly true for a spatial split: a held-out beach
is not a random listing. Coverage on the locked test is therefore a check,
not a theorem. If it misses 80% by a lot, the write-up says so. The radius
is not adjusted after that look.

Quantile regression (pinball loss at 0.1 and 0.9) is the other interval.
It is fit on the development rows and scored once. It is not used to pick
the point model, and LightGBM cannot combine it with monotone constraints,
so the quantile models are unconstrained either way.
"""

from __future__ import annotations

import numpy as np

from evaluation.protocol import INTERVAL_LEVEL


def conformal_radius(abs_log_residual: np.ndarray, level: float = INTERVAL_LEVEL) -> float:
    """Finite-sample split-conformal quantile of absolute log residuals."""
    residual = np.asarray(abs_log_residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    n = len(residual)
    if n == 0:
        raise ValueError("no residuals to calibrate an interval")
    # ceil((n+1)*level)/n, clipped at 1. `higher` matches that order statistic
    # instead of interpolating between two residuals.
    quantile = min(1.0, float(np.ceil((n + 1) * level) / n))
    return float(np.quantile(residual, quantile, method="higher"))


def apply_log_interval(pred_dollars: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Dollar bounds around a dollar prediction, using a log-space radius."""
    pred = np.maximum(np.asarray(pred_dollars, dtype=float), 0.0)
    logged = np.log1p(pred)
    low = np.maximum(np.expm1(logged - radius), 0.0)
    high = np.maximum(np.expm1(logged + radius), 0.0)
    return low, high


def log_abs_residual(y_true: np.ndarray, pred_dollars: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.maximum(np.asarray(pred_dollars, dtype=float), 0.0)
    return np.abs(np.log1p(y_true) - np.log1p(pred))
