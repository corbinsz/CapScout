"""The final spatial holdout, frozen before any v4 model is fit.

Seed 7 was published with v2 and scored again in v3. Seed 11 was the v3
confirmation holdout, and it has been read. This file holds out 20% of the
cells that are in neither, with seed 23. The seed was not searched. Prices
are not in the file. Model selection uses only the cells that remain.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.ingest import load_listings

from evaluation.blocking import confirmation_split, final_split, locked_split
from evaluation.freeze import listing_ids
from evaluation.protocol import (
    BUFFER_KM,
    FINAL_GROUP_FRACTION,
    FINAL_SEED,
    SPATIAL_STEP_DEGREES,
    study_cities,
)

PATH = Path(__file__).resolve().parent / "final_ids.json"


def computed_ids(cities: tuple[str, ...]) -> dict[str, list[str]]:
    frozen: dict[str, list[str]] = {}
    for city in cities:
        frame, _stats = load_listings(city)
        lat = frame["latitude"].to_numpy(dtype=float)
        lon = frame["longitude"].to_numpy(dtype=float)
        hosts = frame["host_id"].to_numpy()
        split = locked_split(lat, lon, hosts)
        confirm = confirmation_split(lat, lon, hosts, split["development_idx"], split["groups"])
        final = final_split(lat, lon, hosts, confirm["pool_idx"], split["groups"])
        ids = listing_ids(frame.iloc[final["final_idx"]]["id"])
        blocked = set(listing_ids(frame.iloc[split["test_idx"]]["id"])) | set(
            listing_ids(frame.iloc[confirm["confirm_idx"]]["id"])
        )
        if set(ids) & blocked:
            raise RuntimeError(f"{city} final holdout overlaps a holdout that was already read")
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{city} final holdout has duplicate ids")
        frozen[city] = sorted(ids, key=int)
    return frozen


def ensure_final_ids(cities: tuple[str, ...] | None = None) -> dict:
    """Write the file once, then refuse to regenerate it."""
    cities = tuple(cities) if cities is not None else study_cities()
    computed = computed_ids(cities)
    if PATH.exists():
        saved = json.loads(PATH.read_text())
        if int(saved.get("final_seed", -1)) != FINAL_SEED:
            raise RuntimeError(
                f"{PATH.name} was frozen at seed {saved.get('final_seed')}, "
                f"but the protocol seed is {FINAL_SEED}."
            )
        for city in cities:
            if saved["cities"].get(city) != computed[city]:
                raise RuntimeError(
                    f"Final holdout ids for {city} do not match seed {FINAL_SEED}. "
                    "The file is a holdout. Do not regenerate it to chase a metric."
                )
        return saved
    payload = {
        "final_seed": FINAL_SEED,
        "spatial_step_degrees": SPATIAL_STEP_DEGREES,
        "final_group_fraction": FINAL_GROUP_FRACTION,
        "buffer_km": BUFFER_KM,
        "drawn_from": (
            "cells in neither the seed-7 locked test nor the seed-11 confirmation holdout"
        ),
        "note": (
            "Frozen before any v4 feature, blend, correction, or interval was fit. "
            "Listing ids only. The split uses coordinates and this seed, not prices. "
            "Seed 23 was not searched."
        ),
        "cities": computed,
    }
    PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
