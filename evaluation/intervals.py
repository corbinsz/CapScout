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


def apply_log_radii(pred_dollars: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Like `apply_log_interval`, with a different radius on every row."""
    pred = np.maximum(np.asarray(pred_dollars, dtype=float), 0.0)
    radii = np.asarray(radii, dtype=float)
    logged = np.log1p(pred)
    low = np.maximum(np.expm1(logged - radii), 0.0)
    high = np.maximum(np.expm1(logged + radii), 0.0)
    return low, high


def fit_log_scale(pred_dollars: np.ndarray, abs_log_residual: np.ndarray) -> tuple[float, float]:
    """Least squares of |log residual| on log1p(prediction).

    A positive slope means expensive predictions are harder in percentage
    terms, which is the Austin failure the flat log band cannot see.
    """
    pred = np.maximum(np.asarray(pred_dollars, dtype=float), 0.0)
    residual = np.asarray(abs_log_residual, dtype=float)
    design = np.column_stack([np.ones(len(pred)), np.log1p(pred)])
    slope, *_rest = np.linalg.lstsq(design, residual, rcond=None)
    return float(slope[0]), float(slope[1])


def sigma_of(pred_dollars: np.ndarray, intercept: float, slope: float, floor: float) -> np.ndarray:
    pred = np.maximum(np.asarray(pred_dollars, dtype=float), 0.0)
    return np.maximum(intercept + slope * np.log1p(pred), floor)


def cqr_nonconformity(
    y_dollars: np.ndarray, q_lo_log: np.ndarray, q_hi_log: np.ndarray
) -> np.ndarray:
    """How far a price sits outside the quantile band, in log space.

    The two quantile predictions are sorted first. A crossed pair would
    otherwise produce a negative width before the conformal step gets a
    chance to repair it. A price inside the band has a negative score,
    and a negative conformal quantile is allowed to shrink the band.
    """
    y_log = np.log1p(np.maximum(np.asarray(y_dollars, dtype=float), 0.0))
    lo = np.minimum(q_lo_log, q_hi_log)
    hi = np.maximum(q_lo_log, q_hi_log)
    return np.maximum(lo - y_log, y_log - hi)


def cqr_bounds(
    q_lo_log: np.ndarray, q_hi_log: np.ndarray, qhat: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dollar interval after expanding the quantile band by `qhat`."""
    lo = np.minimum(q_lo_log, q_hi_log) - qhat
    hi = np.maximum(q_lo_log, q_hi_log) + qhat
    return np.maximum(np.expm1(lo), 0.0), np.maximum(np.expm1(hi), 0.0)


def signed_log_residual(y_true: np.ndarray, pred_dollars: np.ndarray) -> np.ndarray:
    """log1p(actual) minus log1p(predicted). Positive means the model was low."""
    y_true = np.asarray(y_true, dtype=float)
    pred = np.maximum(np.asarray(pred_dollars, dtype=float), 0.0)
    return np.log1p(y_true) - np.log1p(pred)


def asymmetric_quantiles(
    signed_residual: np.ndarray, level: float = INTERVAL_LEVEL
) -> tuple[float, float]:
    """Lower and upper residual quantiles for a two-sided interval.

    The finite-sample levels are the split-conformal ones: the lower tail
    uses the floor order statistic and the upper tail the ceiling one, so
    the band is not shrunk by interpolation. A model that sits low gets a
    longer upper tail without a separate rule for that.
    """
    residual = np.asarray(signed_residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    n = len(residual)
    if n == 0:
        raise ValueError("no residuals to calibrate an interval")
    alpha = 1.0 - float(level)
    lo_level = float(np.floor((n + 1) * (alpha / 2.0)) / n)
    hi_level = float(np.ceil((n + 1) * (1.0 - alpha / 2.0)) / n)
    lo_level = float(np.clip(lo_level, 0.0, 1.0))
    hi_level = float(np.clip(hi_level, 0.0, 1.0))
    if lo_level <= 0.0:
        q_lo = float(np.min(residual))
    else:
        q_lo = float(np.quantile(residual, lo_level, method="lower"))
    if hi_level >= 1.0:
        q_hi = float(np.max(residual))
    else:
        q_hi = float(np.quantile(residual, hi_level, method="higher"))
    return q_lo, q_hi


def apply_signed_interval(
    pred_dollars: np.ndarray, q_lo: np.ndarray | float, q_hi: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Dollar bounds from signed log residuals. The two sides need not match."""
    pred = np.maximum(np.asarray(pred_dollars, dtype=float), 0.0)
    logged = np.log1p(pred)
    low = np.maximum(np.expm1(logged + np.asarray(q_lo, dtype=float)), 0.0)
    high = np.maximum(np.expm1(logged + np.asarray(q_hi, dtype=float)), 0.0)
    return np.minimum(low, high), np.maximum(low, high)


def band_edges(pred_dollars: np.ndarray, n_bands: int) -> np.ndarray:
    """Interior cut points. `n_bands` quintiles need four edges."""
    pred = np.asarray(pred_dollars, dtype=float)
    levels = np.linspace(0.0, 1.0, int(n_bands) + 1)[1:-1]
    return np.quantile(pred, levels)


def assign_band(pred_dollars: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """0 .. len(edges) inclusive. A prediction on an edge goes in the higher band."""
    return np.searchsorted(
        np.asarray(edges, dtype=float), np.asarray(pred_dollars, dtype=float), side="right"
    )


def fit_mondrian(
    pred_dollars: np.ndarray,
    signed_residual: np.ndarray,
    *,
    n_bands: int,
    min_rows: int,
    level: float = INTERVAL_LEVEL,
) -> dict:
    """One asymmetric interval per predicted-price band, else the city-wide one.

    The edges are a function of the calibration predictions, not of the
    prices being covered. A thin band does not get its own quantile.
    """
    edges = band_edges(pred_dollars, n_bands)
    bands = assign_band(pred_dollars, edges)
    residual = np.asarray(signed_residual, dtype=float)
    global_lo, global_hi = asymmetric_quantiles(residual, level)
    per_band = []
    counts = []
    for band in range(int(n_bands)):
        mask = bands == band
        count = int(mask.sum())
        counts.append(count)
        if count >= int(min_rows):
            per_band.append(list(asymmetric_quantiles(residual[mask], level)))
        else:
            per_band.append([global_lo, global_hi])
    return {
        "edges": [float(edge) for edge in edges],
        "bands": per_band,
        "counts": counts,
        "global": [global_lo, global_hi],
        "n_bands": int(n_bands),
        "min_rows": int(min_rows),
    }


def apply_mondrian(pred_dollars: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply band-wise residual quantiles. Edges come from calibration predictions."""
    edges = np.asarray(params["edges"], dtype=float)
    bands = assign_band(pred_dollars, edges)
    q_lo = np.full(len(np.asarray(pred_dollars)), np.nan, dtype=float)
    q_hi = np.full(len(q_lo), np.nan, dtype=float)
    for band, (lo, hi) in enumerate(params["bands"]):
        mask = bands == band
        q_lo[mask] = float(lo)
        q_hi[mask] = float(hi)
    # A prediction above every calibration edge is the last band. searchsorted
    # already returns len(edges), which is the last band index when there are
    # len(edges)+1 bands. If a band was never filled, fall back to the global pair.
    missing = ~np.isfinite(q_lo)
    if missing.any():
        q_lo[missing] = float(params["global"][0])
        q_hi[missing] = float(params["global"][1])
    return apply_signed_interval(pred_dollars, q_lo, q_hi)
