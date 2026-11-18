"""Checks that can be wrong without the metrics looking wrong.

These do not decide the model. They record, for the write-up, that the
locked test still has no shared host and no neighbour inside the buffer,
and that the scored columns are the ones the protocol resolved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from features.columns import KNN_COLUMNS, SMOOTH_COLUMNS

from evaluation.blocking import buffer_train, purge_hosts
from evaluation.protocol import BUFFER_KM, SPATIAL_STEP_DEGREES
from evaluation.splits import spatial_group_ids


def split_audit(
    frame: pd.DataFrame,
    development_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    lat = frame["latitude"].to_numpy(dtype=float)
    lon = frame["longitude"].to_numpy(dtype=float)
    hosts = frame["host_id"].to_numpy()
    groups = spatial_group_ids(lat, lon, SPATIAL_STEP_DEGREES)
    shared_cells = set(groups[development_idx]) & set(groups[test_idx])
    dev_hosts = set(hosts[development_idx])
    test_hosts = set(hosts[test_idx])
    _kept, n_close = buffer_train(lat, lon, development_idx, test_idx, BUFFER_KM)
    # Re-applying the host purge to the development rows must remove nobody:
    # they were already removed when the split was built.
    _again, n_hosts = purge_hosts(hosts, development_idx, test_idx)
    if len(development_idx) and len(test_idx):
        from sklearn.neighbors import BallTree

        tree = BallTree(
            np.radians(np.column_stack([lat[test_idx], lon[test_idx]])),
            metric="haversine",
        )
        dist, _idx = tree.query(
            np.radians(np.column_stack([lat[development_idx], lon[development_idx]])),
            k=1,
        )
        min_km = float(np.min(dist) * 6371.0)
    else:
        min_km = None
    return {
        "shared_cells": int(len(shared_cells)),
        "shared_hosts": int(len(dev_hosts & test_hosts)),
        "hosts_still_to_purge": int(n_hosts),
        "training_rows_inside_buffer": int(n_close),
        "min_train_test_km": min_km,
        "passed": (
            len(shared_cells) == 0
            and len(dev_hosts & test_hosts) == 0
            and n_hosts == 0
            and n_close == 0
            and (min_km is None or min_km >= BUFFER_KM - 1e-6)
        ),
    }


def column_audit(columns: list[str]) -> dict:
    """A price column that is not a declared fold-fit encoding is a leak."""
    allowed_price = set(SMOOTH_COLUMNS) | set(KNN_COLUMNS)
    suspicious = [
        column
        for column in columns
        if "price" in column or column in {"y", "price_clean", "occupancy_rate"}
        if column not in allowed_price
    ]
    return {
        "n_columns": len(columns),
        "fold_fit_price_columns": [column for column in columns if column in allowed_price],
        "unexpected_price_columns": suspicious,
        "passed": len(suspicious) == 0,
    }
