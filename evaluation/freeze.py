"""The locked test, identified by listing id.

The ids are a function of coordinates, the host ids (only for the training
side), and TEST_SEED. They do not depend on Census, OpenStreetMap, or any
model. The file is written the first time and then checked. A mismatch
raises. Regenerating the file to match a new seed would be a new study,
and this module will not do that quietly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from data.ingest import load_listings

from evaluation.blocking import locked_split
from evaluation.protocol import (
    BUFFER_KM,
    SPATIAL_STEP_DEGREES,
    TEST_GROUP_FRACTION,
    TEST_SEED,
    study_cities,
)

PATH = Path(__file__).resolve().parent / "locked_test_ids.json"


def listing_ids(series: pd.Series) -> list[str]:
    """Inside Airbnb ids are integers. Force that, so '123.0' cannot sneak in."""
    values = pd.to_numeric(series, errors="raise").astype("int64")
    return [str(value) for value in values.tolist()]


def _computed(cities: tuple[str, ...]) -> dict[str, list[str]]:
    frozen: dict[str, list[str]] = {}
    for city in cities:
        frame, _stats = load_listings(city)
        split = locked_split(
            frame["latitude"].to_numpy(dtype=float),
            frame["longitude"].to_numpy(dtype=float),
            frame["host_id"].to_numpy(),
        )
        ids = listing_ids(frame.iloc[split["test_idx"]]["id"])
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{city} locked test has duplicate listing ids")
        frozen[city] = sorted(ids, key=int)
    return frozen


def ensure_frozen_ids(cities: tuple[str, ...] | None = None) -> dict:
    cities = tuple(cities) if cities is not None else study_cities()
    computed = _computed(cities)
    if PATH.exists():
        saved = json.loads(PATH.read_text())
        if int(saved.get("test_seed", -1)) != TEST_SEED:
            raise RuntimeError(
                f"{PATH.name} was frozen at seed {saved.get('test_seed')}, "
                f"but the protocol seed is {TEST_SEED}."
            )
        for city in cities:
            if saved["cities"].get(city) != computed[city]:
                raise RuntimeError(
                    f"Frozen test ids for {city} do not match seed {TEST_SEED}. "
                    "The file is the holdout. Do not regenerate it to chase a metric."
                )
        return saved
    payload = {
        "test_seed": TEST_SEED,
        "spatial_step_degrees": SPATIAL_STEP_DEGREES,
        "test_group_fraction": TEST_GROUP_FRACTION,
        "buffer_km": BUFFER_KM,
        "note": (
            "Frozen before Census and OpenStreetMap features were scored. "
            "Listing ids only. The split uses coordinates and this seed, not prices."
        ),
        "cities": computed,
    }
    PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
