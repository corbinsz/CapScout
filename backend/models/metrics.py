"""Honest regression metrics and split helpers.

These functions do not train anything. The benchmark and the tests both use them
so a "good" R-squared cannot be invented by scoring a model on rows it just fit.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'n': 0, 'mae': None, 'rmse': None, 'r2': None, 'mape_pct': None}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = None
    if len(y_true) > 1 and float(np.var(y_true)) > 1e-8:
        r2 = float(r2_score(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs(y_pred - y_true) / denom) * 100)
    return {
        'n': int(len(y_true)),
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': None if r2 is None else round(r2, 4),
        'mape_pct': round(mape, 2),
    }


def group_holdout(listing_ids, test_size=0.2, seed=42):
    """Hold out whole listings. A listing never sits in both train and test."""
    groups = np.asarray(listing_ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dummy = np.zeros(len(groups))
    train_idx, test_idx = next(splitter.split(dummy, groups=groups))
    return train_idx, test_idx


def time_holdout(timestamps, test_fraction=0.2):
    """Train on older rows and test on the newest rows.

    Rows with a missing timestamp stay in train. Returns None when the newest
    slice would be too small to mean anything.
    """
    stamps = np.asarray(timestamps)
    known = np.array([stamp == stamp and stamp is not None for stamp in stamps])
    # NaT / NaN: pandas timestamps use NaT which fails equality with itself.
    if np.issubdtype(stamps.dtype, np.datetime64):
        known = ~np.isnat(stamps)
    if int(known.sum()) < 50:
        return None
    order = np.argsort(stamps[known].astype('datetime64[ns]'))
    known_idx = np.flatnonzero(known)[order]
    cut = int(len(known_idx) * (1 - test_fraction))
    if cut < 30 or len(known_idx) - cut < 30:
        return None
    test_idx = known_idx[cut:]
    train_idx = np.setdiff1d(np.arange(len(stamps)), test_idx, assume_unique=False)
    return train_idx, test_idx
