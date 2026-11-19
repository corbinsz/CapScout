"""Spatial holdout for the nights study.

Seed 31, 20% of cells, on listings that have an observed booked-nights
label. The file stores listing ids only. It is written once and then
checked. A mismatch raises. The label is not in the file, and the seed
is not a value that was searched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from data.ingest import load_listings

from evaluation.blocking import protect_train
from evaluation.freeze import listing_ids
from evaluation.occupancy_protocol import OCCUPANCY_GROUP_FRACTION, OCCUPANCY_SEED
from evaluation.protocol import BUFFER_KM, SPATIAL_STEP_DEGREES
from evaluation.splits import holdout_groups, spatial_group_ids

PATH = Path(__file__).resolve().parent / "occupancy_ids.json"


def population(frame: pd.DataFrame) -> pd.DataFrame:
    """Listings with an observed nights label. Order is the file order."""
    nights = pd.to_numeric(frame["booked_nights"], errors="coerce").to_numpy(dtype=float)
    return frame.loc[np.isfinite(nights)].reset_index(drop=True)


def occupancy_split(lat: np.ndarray, lon: np.ndarray, hosts: np.ndarray) -> dict:
    """One spatial holdout. Seed and fraction come from the occupancy protocol."""
    groups = spatial_group_ids(lat, lon, SPATIAL_STEP_DEGREES)
    train_idx, test_idx = holdout_groups(groups, OCCUPANCY_GROUP_FRACTION, OCCUPANCY_SEED)
    dev_idx, info = protect_train(lat, lon, hosts, train_idx, test_idx)
    info.update(
        {
            "n_rows": int(len(lat)),
            "n_groups": int(len(np.unique(groups))),
            "n_test": int(len(test_idx)),
            "test_fraction_of_rows": float(len(test_idx) / max(len(lat), 1)),
            "occupancy_seed": OCCUPANCY_SEED,
        }
    )
    return {
        "groups": groups,
        "development_idx": dev_idx,
        "test_idx": test_idx,
        "info": info,
    }


def computed_ids(cities: tuple[str, ...]) -> dict[str, list[str]]:
    frozen: dict[str, list[str]] = {}
    for city in cities:
        frame, _stats = load_listings(city)
        observed = population(frame)
        split = occupancy_split(
            observed["latitude"].to_numpy(dtype=float),
            observed["longitude"].to_numpy(dtype=float),
            observed["host_id"].to_numpy(),
        )
        ids = listing_ids(observed.iloc[split["test_idx"]]["id"])
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{city} occupancy holdout has duplicate listing ids")
        frozen[city] = sorted(ids, key=int)
    return frozen


def ensure_occupancy_ids(cities: tuple[str, ...]) -> dict:
    """Write the file once, then refuse to regenerate it."""
    computed = computed_ids(cities)
    if PATH.exists():
        saved = json.loads(PATH.read_text())
        if int(saved.get("occupancy_seed", -1)) != OCCUPANCY_SEED:
            raise RuntimeError(
                f"{PATH.name} was frozen at seed {saved.get('occupancy_seed')}, "
                f"but the protocol seed is {OCCUPANCY_SEED}."
            )
        for city in cities:
            if saved["cities"].get(city) != computed[city]:
                raise RuntimeError(
                    f"Occupancy holdout ids for {city} do not match seed {OCCUPANCY_SEED}. "
                    "The file is a holdout. Do not regenerate it to chase a metric."
                )
        return saved
    payload = {
        "occupancy_seed": OCCUPANCY_SEED,
        "spatial_step_degrees": SPATIAL_STEP_DEGREES,
        "test_group_fraction": OCCUPANCY_GROUP_FRACTION,
        "buffer_km": BUFFER_KM,
        "population": (
            "Cleaned listings with at least one review in the last 12 months. "
            "A row with no reviews in that window is unobserved, not zero nights."
        ),
        "note": (
            "Frozen before any occupancy feature set or model was fit. "
            "Listing ids only. The split uses coordinates and seed 31, not the "
            "nights label and not the price. Seed 31 was not searched. It is "
            "not 0, 1, 2, 3, 4, 7, 11, 23, or 42."
        ),
        "cities": computed,
    }
    PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
