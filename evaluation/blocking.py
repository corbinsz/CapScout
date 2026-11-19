"""Spatial holdout, host purge, and a 1 km buffer.

Group ids are a function of latitude and longitude only. Host ids and
distances are applied after the split, and only to the training side, so
the test set stays the cells that were held out.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import BallTree

from evaluation.protocol import (
    BUFFER_KM,
    CONFIRM_GROUP_FRACTION,
    CONFIRM_SEED,
    FINAL_GROUP_FRACTION,
    FINAL_SEED,
    INNER_GROUP_FRACTION,
    INNER_SEEDS,
    SPATIAL_STEP_DEGREES,
    TEST_GROUP_FRACTION,
    TEST_SEED,
)
from evaluation.splits import holdout_groups, spatial_group_ids

EARTH_RADIUS_KM = 6371.0


def _radians(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.radians(np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)]))


def purge_hosts(
    hosts: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """Drop training rows whose host also has a row on the other side."""
    blocked = set(np.asarray(hosts)[test_idx])
    keep = np.array([i for i in train_idx if hosts[i] not in blocked], dtype=int)
    return keep, int(len(train_idx) - len(keep))


def buffer_train(
    lat: np.ndarray,
    lon: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    buffer_km: float = BUFFER_KM,
) -> tuple[np.ndarray, int]:
    """Drop training rows within buffer_km of any held-out row."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return train_idx, 0
    tree = BallTree(_radians(lat[test_idx], lon[test_idx]), metric="haversine")
    dist, _idx = tree.query(_radians(lat[train_idx], lon[train_idx]), k=1)
    dist_km = np.atleast_1d(dist).ravel() * EARTH_RADIUS_KM
    close = dist_km < buffer_km
    keep = train_idx[~close]
    return keep, int(close.sum())


def protect_train(
    lat: np.ndarray,
    lon: np.ndarray,
    hosts: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    after_hosts, n_hosts = purge_hosts(hosts, train_idx, test_idx)
    after_buffer, n_buffer = buffer_train(lat, lon, after_hosts, test_idx)
    return after_buffer, {
        "train_before": int(len(train_idx)),
        "removed_host": n_hosts,
        "removed_buffer": n_buffer,
        "train_after": int(len(after_buffer)),
    }


def locked_split(lat: np.ndarray, lon: np.ndarray, hosts: np.ndarray) -> dict:
    """The one test split. Seed and fraction come from the protocol."""
    groups = spatial_group_ids(lat, lon, SPATIAL_STEP_DEGREES)
    index = np.arange(len(lat))
    train_idx, test_idx = holdout_groups(groups, TEST_GROUP_FRACTION, TEST_SEED)
    # holdout_groups returns positions into `groups`, which is every row.
    dev_idx, info = protect_train(lat, lon, hosts, train_idx, test_idx)
    info.update(
        {
            "n_rows": int(len(lat)),
            "n_groups": int(len(np.unique(groups))),
            "n_test": int(len(test_idx)),
            "test_fraction_of_rows": float(len(test_idx) / len(lat)),
        }
    )
    return {
        "groups": groups,
        "development_idx": dev_idx,
        "test_idx": test_idx,
        "info": info,
        "index": index,
    }


def confirmation_split(
    lat: np.ndarray,
    lon: np.ndarray,
    hosts: np.ndarray,
    development_idx: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Hold out development cells for the confirmation check.

    The locked test is not in `development_idx`, so this split cannot
    touch it. `pool_idx` is what model selection is allowed to see.
    `train_idx` is that pool after the host purge and the 1 km buffer
    against the confirmation cells, which is the training set for the
    one confirmation score.
    """
    dev = np.asarray(development_idx, dtype=int)
    rel_pool, rel_confirm = holdout_groups(groups[dev], CONFIRM_GROUP_FRACTION, CONFIRM_SEED)
    confirm_idx = dev[rel_confirm]
    pool_idx = dev[rel_pool]
    train_idx, info = protect_train(lat, lon, hosts, pool_idx, confirm_idx)
    info.update(
        {
            "n_confirm": int(len(confirm_idx)),
            "n_pool": int(len(pool_idx)),
            "confirm_seed": CONFIRM_SEED,
            "confirm_group_fraction": CONFIRM_GROUP_FRACTION,
        }
    )
    return {
        "confirm_idx": confirm_idx,
        "pool_idx": pool_idx,
        "train_idx": train_idx,
        "info": info,
    }


def final_split(
    lat: np.ndarray,
    lon: np.ndarray,
    hosts: np.ndarray,
    pool_idx: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Hold out cells that are in neither the locked test nor the confirmation set.

    `selection_idx` is what model selection is allowed to see. `train_idx`
    is that pool after the host purge and the 1 km buffer against the final
    cells. The model that scores the final holdout is trained on a wider
    set (see `score_split`); this training index is the selection pool only.
    """
    pool = np.asarray(pool_idx, dtype=int)
    rel_keep, rel_final = holdout_groups(groups[pool], FINAL_GROUP_FRACTION, FINAL_SEED)
    final_idx = pool[rel_final]
    selection_idx = pool[rel_keep]
    train_idx, info = protect_train(lat, lon, hosts, selection_idx, final_idx)
    info.update(
        {
            "n_final": int(len(final_idx)),
            "n_selection": int(len(selection_idx)),
            "final_seed": FINAL_SEED,
            "final_group_fraction": FINAL_GROUP_FRACTION,
        }
    )
    return {
        "final_idx": final_idx,
        "selection_idx": selection_idx,
        "train_idx": train_idx,
        "info": info,
    }


def score_split(
    lat: np.ndarray,
    lon: np.ndarray,
    hosts: np.ndarray,
    holdout_idx: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Every row outside the holdout's cells, then the host purge and the buffer.

    A cell is held out whole. Listings in that cell which were already
    removed from an earlier pool (the 1 km buffer around a different
    holdout) are not training rows either. Leaving them in would put the
    same block on both sides. Selection does not use this function. It uses
    `final_split`'s selection pool, which also excludes the other two holdouts.
    """
    holdout = np.asarray(holdout_idx, dtype=int)
    groups = spatial_group_ids(lat, lon, SPATIAL_STEP_DEGREES)
    blocked = set(groups[holdout])
    pool = np.array([i for i in range(len(lat)) if groups[i] not in blocked], dtype=int)
    train_idx, info = protect_train(lat, lon, hosts, pool, holdout)
    info["n_holdout"] = int(len(holdout))
    return train_idx, info


def development_splits(
    groups: np.ndarray,
    development_idx: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    hosts: np.ndarray,
    seeds: tuple[int, ...] = INNER_SEEDS,
):
    """Repeated spatial holdouts of the development set only."""
    dev = np.asarray(development_idx, dtype=int)
    dev_groups = groups[dev]
    for seed in seeds:
        rel_train, rel_valid = holdout_groups(dev_groups, INNER_GROUP_FRACTION, seed)
        train_idx = dev[rel_train]
        valid_idx = dev[rel_valid]
        train_idx, info = protect_train(lat, lon, hosts, train_idx, valid_idx)
        if len(train_idx) < 80 or len(valid_idx) < 30:
            continue
        yield seed, train_idx, valid_idx, info
