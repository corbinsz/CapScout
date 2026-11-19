"""Spatial cells, the frozen holdout, and the training-side buffer.

A cell is 3 km on a side in a local projection. The projection uses the
city's reference latitude, so the cell size does not depend on price.
Twenty percent of the cells are held out. Every listing in a held-out cell
is held out. A host who has a held-out listing loses their other listings
from the training side. A training listing within 1 km of a held-out
listing is dropped. The test side is not trimmed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from data.cities import CITIES, STUDY_CITIES
from features.geo import distance_to_city_center_km

from evaluation.blocking import buffer_train, protect_train
from evaluation.freeze import listing_ids
from evaluation.study_protocol import (
    BUFFER_KM,
    CELL_KM,
    CONFIRM_SEEDS,
    HOLDOUT_FRACTION,
    MAX_KM_FROM_CENTER,
    PRIMARY_SEED,
    assert_seeds,
)

PATH = Path(__file__).resolve().parent / "study_ids.json"
METERS_PER_DEGREE = 111_320.0


def within_city(lat: float, lon: float, city_key: str, max_km: float = MAX_KM_FROM_CENTER) -> bool:
    distance = distance_to_city_center_km(float(lat), float(lon), city_key)
    return bool(np.asarray(distance).reshape(-1)[0] <= max_km)


def cell_ids(
    lat: np.ndarray, lon: np.ndarray, city_key: str, cell_km: float = CELL_KM
) -> np.ndarray:
    """Integer cell id from coordinates alone. About `cell_km` on a side."""
    lat0 = float(CITIES[city_key]["latitude"])
    cell_m = float(cell_km) * 1000.0
    y = np.floor(np.asarray(lat, dtype=float) * METERS_PER_DEGREE / cell_m).astype(np.int64)
    x_scale = METERS_PER_DEGREE * np.cos(np.radians(lat0))
    x = np.floor(np.asarray(lon, dtype=float) * x_scale / cell_m).astype(np.int64)
    return (x + 100_000) * 1_000_000 + (y + 100_000)


def choose_test_cells(cells: np.ndarray, fraction: float, seed: int) -> set[int]:
    """Hold out `fraction` of the distinct cells. The cell list is sorted first."""
    unique = np.array(sorted({int(cell) for cell in np.asarray(cells)}), dtype=np.int64)
    if len(unique) < 2:
        raise RuntimeError("need at least two spatial cells")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(unique))
    n_test = int(round(len(unique) * float(fraction)))
    n_test = min(max(n_test, 1), len(unique) - 1)
    return set(int(cell) for cell in unique[order[:n_test]])


def make_split(
    lat: np.ndarray,
    lon: np.ndarray,
    hosts: np.ndarray,
    cells: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> dict:
    """Test indexes are the held-out cells. Training indexes are buffered."""
    test_cells = choose_test_cells(cells, fraction, seed)
    test_mask = np.isin(np.asarray(cells), list(test_cells))
    test_idx = np.flatnonzero(test_mask)
    train_pool = np.flatnonzero(~test_mask)
    train_idx, info = protect_train(
        np.asarray(lat, dtype=float),
        np.asarray(lon, dtype=float),
        np.asarray(hosts),
        train_pool,
        test_idx,
    )
    info["n_test_cells"] = int(len(test_cells))
    info["n_train_cells"] = int(len(set(int(cell) for cell in np.asarray(cells)[train_idx])))
    info["n_test"] = int(len(test_idx))
    info["buffer_km"] = BUFFER_KM
    return {"train_idx": train_idx, "test_idx": test_idx, "info": info}


def audit_split(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """No shared cell, no shared host, and nothing inside the 1 km buffer."""
    cells = frame["cell"].to_numpy()
    hosts = frame["host_id"].astype(str).to_numpy()
    shared_cells = set(cells[train_idx].tolist()) & set(cells[test_idx].tolist())
    shared_hosts = set(hosts[train_idx].tolist()) & set(hosts[test_idx].tolist())
    lat = frame["latitude"].to_numpy(dtype=float)
    lon = frame["longitude"].to_numpy(dtype=float)
    _kept, n_close = buffer_train(lat, lon, np.asarray(train_idx), np.asarray(test_idx), BUFFER_KM)
    if len(train_idx) and len(test_idx):
        from sklearn.neighbors import BallTree

        tree = BallTree(
            np.radians(np.column_stack([lat[test_idx], lon[test_idx]])),
            metric="haversine",
        )
        dist, _idx = tree.query(
            np.radians(np.column_stack([lat[train_idx], lon[train_idx]])),
            k=1,
        )
        min_km = float(np.min(dist) * 6371.0)
    else:
        min_km = None
    passed = (
        len(shared_cells) == 0
        and len(shared_hosts) == 0
        and int(n_close) == 0
        and (min_km is None or min_km >= BUFFER_KM - 1e-6)
    )
    return {
        "shared_cells": int(len(shared_cells)),
        "shared_hosts": int(len(shared_hosts)),
        "training_rows_inside_buffer": int(n_close),
        "min_train_test_km": min_km,
        "passed": passed,
    }


def split_frame(frame: pd.DataFrame, city_key: str, seed: int) -> dict:
    return make_split(
        frame["latitude"].to_numpy(dtype=float),
        frame["longitude"].to_numpy(dtype=float),
        frame["host_id"].to_numpy(),
        frame["cell"].to_numpy(),
        fraction=HOLDOUT_FRACTION,
        seed=seed,
    )


def _city_ids(frame: pd.DataFrame, city_key: str, seed: int) -> list[str]:
    split = split_frame(frame, city_key, seed)
    ids = listing_ids(frame.iloc[split["test_idx"]]["id"])
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{city_key} seed {seed} has duplicate listing ids")
    return sorted(ids, key=int)


def computed_ids(frames: dict[str, pd.DataFrame]) -> dict:
    assert_seeds()
    cities = {}
    for city in STUDY_CITIES:
        frame = frames[city]
        cities[city] = {
            "primary": _city_ids(frame, city, PRIMARY_SEED),
            "confirm": {str(seed): _city_ids(frame, city, seed) for seed in CONFIRM_SEEDS},
        }
    return {
        "primary_seed": PRIMARY_SEED,
        "confirm_seeds": list(CONFIRM_SEEDS),
        "cell_km": CELL_KM,
        "fraction": HOLDOUT_FRACTION,
        "buffer_km": BUFFER_KM,
        "max_km_from_center": MAX_KM_FROM_CENTER,
        "note": (
            "Listing ids only. Computed from coordinates and these seeds "
            "before any model was fit. The file contains no prices."
        ),
        "cities": cities,
    }


def ensure_ids(frames: dict[str, pd.DataFrame]) -> dict:
    """Write the id file the first time. After that, refuse a mismatch."""
    computed = computed_ids(frames)
    if PATH.exists():
        saved = json.loads(PATH.read_text())
        if int(saved.get("primary_seed", -1)) != PRIMARY_SEED:
            raise RuntimeError(
                f"{PATH.name} was frozen at seed {saved.get('primary_seed')}, "
                f"not {PRIMARY_SEED}."
            )
        if [int(seed) for seed in saved.get("confirm_seeds", [])] != list(CONFIRM_SEEDS):
            raise RuntimeError(f"{PATH.name} confirmation seeds do not match the protocol.")
        for city in STUDY_CITIES:
            if saved["cities"][city] != computed["cities"][city]:
                raise RuntimeError(
                    f"Frozen ids for {city} do not match the protocol. "
                    "Do not regenerate them to chase a metric."
                )
        return saved
    PATH.write_text(json.dumps(computed, indent=2) + "\n")
    return computed
