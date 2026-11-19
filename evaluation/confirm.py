"""A second spatial holdout, frozen before any v3 model is fit.

Seed 7 has been published. Using it to choose features would be tuning on
a test that has already been read. This file holds out 20% of the
*development* cells, with seed 11, so the new features, the pooled model,
and the interval method can be chosen without those listings and then
checked once. The seed-7 listings are not in this file. They stay the
same-rows comparison against v2.

The ids are a function of coordinates and the seed. Prices are not in here.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.ingest import load_listings

from evaluation.blocking import confirmation_split, locked_split
from evaluation.freeze import listing_ids
from evaluation.protocol import (
    BUFFER_KM,
    CONFIRM_GROUP_FRACTION,
    CONFIRM_SEED,
    SPATIAL_STEP_DEGREES,
    study_cities,
)

PATH = Path(__file__).resolve().parent / "confirm_ids.json"


def computed_ids(cities: tuple[str, ...]) -> dict[str, list[str]]:
    frozen: dict[str, list[str]] = {}
    for city in cities:
        frame, _stats = load_listings(city)
        lat = frame["latitude"].to_numpy(dtype=float)
        lon = frame["longitude"].to_numpy(dtype=float)
        hosts = frame["host_id"].to_numpy()
        split = locked_split(lat, lon, hosts)
        confirm = confirmation_split(lat, lon, hosts, split["development_idx"], split["groups"])
        ids = listing_ids(frame.iloc[confirm["confirm_idx"]]["id"])
        test_ids = set(listing_ids(frame.iloc[split["test_idx"]]["id"]))
        if set(ids) & test_ids:
            raise RuntimeError(f"{city} confirmation holdout overlaps the locked test")
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{city} confirmation holdout has duplicate ids")
        frozen[city] = sorted(ids, key=int)
    return frozen


def ensure_confirm_ids(cities: tuple[str, ...] | None = None) -> dict:
    """Write the file once, then refuse to regenerate it."""
    cities = tuple(cities) if cities is not None else study_cities()
    computed = computed_ids(cities)
    if PATH.exists():
        saved = json.loads(PATH.read_text())
        if int(saved.get("confirm_seed", -1)) != CONFIRM_SEED:
            raise RuntimeError(
                f"{PATH.name} was frozen at seed {saved.get('confirm_seed')}, "
                f"but the protocol seed is {CONFIRM_SEED}."
            )
        for city in cities:
            if saved["cities"].get(city) != computed[city]:
                raise RuntimeError(
                    f"Confirmation ids for {city} do not match seed {CONFIRM_SEED}. "
                    "The file is a holdout. Do not regenerate it to chase a metric."
                )
        return saved
    payload = {
        "confirm_seed": CONFIRM_SEED,
        "spatial_step_degrees": SPATIAL_STEP_DEGREES,
        "confirm_group_fraction": CONFIRM_GROUP_FRACTION,
        "buffer_km": BUFFER_KM,
        "drawn_from": "development cells only, after the seed-7 locked test was removed",
        "note": (
            "Frozen before any v3 feature, pooled model, or interval was fit. "
            "Listing ids only. The split uses coordinates and this seed, not prices. "
            "Seed 11 was not searched."
        ),
        "cities": computed,
    }
    PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
