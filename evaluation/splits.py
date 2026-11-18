"""Spatial blocking.

A random 20% of listings still trains on the house next door. Advertised
rates are spatially smooth, so that split overstates how well the model
would do on a block it has not seen. GroupKFold on a 0.03-degree grid
(about 3 km of latitude) holds out every listing in a cell together.

The grid is a property of the coordinates, not of the prices, so it can
be assigned before the split. 0.03 degrees is coarse enough that a cell
is a neighbourhood-scale block and fine enough that a city still has many
cells. GroupKFold then distributes cells, not rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def spatial_group_ids(lat: np.ndarray, lon: np.ndarray, step: float = 0.03) -> np.ndarray:
    """Cell id for each listing. Same coordinates always share a cell."""
    lat_bin = np.round(np.asarray(lat, dtype=float) / step) * step
    lon_bin = np.round(np.asarray(lon, dtype=float) / step) * step
    return np.array([f"{a:.3f}|{b:.3f}" for a, b in zip(lat_bin, lon_bin, strict=True)])


def iter_group_folds(groups: np.ndarray, n_splits: int = 5):
    """Yield (train_idx, test_idx) with no group in both sides.

    Fewer splits are used when a city does not have enough distinct cells.
    """
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        raise ValueError(f"Need at least 2 spatial groups, found {n_groups}")
    folder = GroupKFold(n_splits=splits)
    dummy = np.zeros(len(groups))
    yield from folder.split(dummy, dummy, groups)


def holdout_groups(groups: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """One group-aware split. Used for the inner early-stopping cut."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dummy = np.zeros(len(groups))
    train_idx, valid_idx = next(splitter.split(dummy, dummy, groups))
    return train_idx, valid_idx
