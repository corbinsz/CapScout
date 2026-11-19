"""Fixed listing panel for the daily calendar collector.

The occupancy model is weak because its label is a review-count proxy.
Replacing that label needs the same homes watched every day. This module
draws that panel once, from the Inside Airbnb listings already in the repo,
stratified by neighbourhood and room type. The daily job reads the committed
ids. It does not redraw them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from data.cities import STUDY_CITIES
from data.ingest import listings_path

# Not a modeling seed. The price study used 0–4, 7, 11, 23, and 42.
# The nights study used 31. This draw is only a panel, not a holdout.
SAMPLE_SEED = 17
SAMPLE_SIZE = 300
SAMPLE_CITIES: tuple[str, ...] = STUDY_CITIES
REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "data" / "calendar_sample.json"


def listing_id(value: object) -> str | None:
    """Inside Airbnb ids arrive as ints, floats, or strings. Keep one form."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _stratum_label(neighbourhood: object, room_type: object) -> str:
    hood = str(neighbourhood).strip() if pd.notna(neighbourhood) else ""
    room = str(room_type).strip() if pd.notna(room_type) else ""
    return f"{hood or 'unknown'}|{room or 'unknown'}"


def load_strata(city_key: str, path: Path | None = None) -> pd.DataFrame:
    """One row per listing id, with the stratum used for the draw."""
    frame = pd.read_csv(
        path or listings_path(city_key),
        usecols=["id", "neighbourhood_cleansed", "room_type"],
    )
    frame["listing_id"] = frame["id"].map(listing_id)
    frame = frame[frame["listing_id"].notna()].drop_duplicates("listing_id")
    frame["stratum"] = [
        _stratum_label(hood, room)
        for hood, room in zip(frame["neighbourhood_cleansed"], frame["room_type"], strict=True)
    ]
    return frame[["listing_id", "stratum"]].reset_index(drop=True)


def allocate_strata(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation, capped at each stratum's size.

    Strata are visited in sorted name order on ties, so the result does not
    depend on row order. A stratum can receive zero when the panel is smaller
    than the number of strata. The draw then follows the larger strata.
    """
    keys = sorted(counts)
    sizes = {key: int(counts[key]) for key in keys}
    total = sum(sizes.values())
    if n <= 0 or total <= 0:
        return {key: 0 for key in keys}
    if n >= total:
        return dict(sizes)

    raw = {key: sizes[key] * (n / total) for key in keys}
    alloc = {key: int(np.floor(raw[key])) for key in keys}
    short = n - sum(alloc.values())
    order = sorted(keys, key=lambda key: (-(raw[key] - alloc[key]), key))
    for key in order:
        if short <= 0:
            break
        if alloc[key] < sizes[key]:
            alloc[key] += 1
            short -= 1
    if short > 0:
        for key in sorted(keys, key=lambda key: (-sizes[key], key)):
            while short > 0 and alloc[key] < sizes[key]:
                alloc[key] += 1
                short -= 1
    return alloc


def _id_sort_key(value: str) -> tuple:
    return (0, int(value)) if value.isdigit() else (1, value)


def draw_ids(
    frame: pd.DataFrame,
    n: int,
    seed: int,
    city_index: int,
) -> list[str]:
    """Seeded without-replacement draw inside the allocated strata.

    ``city_index`` is part of the seed, so editing another city does not
    reshuffle this one. Pools are sorted before the draw, and the returned
    ids are sorted, so file order cannot change the panel.
    """
    counts: dict[str, int] = {}
    pools: dict[str, list[str]] = {}
    for listing, stratum in zip(frame["listing_id"], frame["stratum"], strict=True):
        counts[stratum] = counts.get(stratum, 0) + 1
        pools.setdefault(stratum, []).append(str(listing))
    alloc = allocate_strata(counts, n)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(city_index)]))
    chosen: list[str] = []
    for stratum in sorted(alloc):
        take = alloc[stratum]
        if take <= 0:
            continue
        pool = np.array(sorted(pools[stratum], key=_id_sort_key), dtype=object)
        picked = rng.choice(pool, size=int(take), replace=False)
        chosen.extend(str(item) for item in picked)
    return sorted(chosen, key=_id_sort_key)


def build_sample_document(
    cities: Sequence[str] = SAMPLE_CITIES,
    per_city: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> dict:
    """The document committed at ``data/calendar_sample.json``."""
    payload: dict = {
        "seed": seed,
        "per_city": per_city,
        "strata": ["neighbourhood_cleansed", "room_type"],
        "allocation": "largest remainder, capped at the stratum size",
        "draw": "SeedSequence([seed, city_index]); pools sorted; ids sorted",
        "cities": {},
    }
    for index, city in enumerate(cities):
        frame = load_strata(city)
        ids = draw_ids(frame, per_city, seed, index)
        payload["cities"][city] = {
            "source_rows": int(len(frame)),
            "strata": int(frame["stratum"].nunique()),
            "sampled": len(ids),
            "listing_ids": ids,
        }
    return payload


def write_sample(path: Path | None = None) -> dict:
    document = build_sample_document()
    dest = path or SAMPLE_PATH
    dest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def load_sample(path: Path | None = None) -> dict[str, list[str]]:
    """City key to the committed listing ids, as strings."""
    document = json.loads((path or SAMPLE_PATH).read_text(encoding="utf-8"))
    return {
        city: [str(listing_id) for listing_id in body["listing_ids"]]
        for city, body in document["cities"].items()
    }
