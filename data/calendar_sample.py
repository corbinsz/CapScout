"""Fixed listing panel for the daily calendar collector.

The occupancy model is weak because its label is a review-count proxy.
Replacing that label needs the same homes watched every day. This module
draws that panel once, from the newest Inside Airbnb listings, keeping homes
with a review in the twelve months before that file and at least one open
night. The draw is large enough that about 300 of those calendars come back.
The daily job reads the committed ids. It does not redraw them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from data.cities import STUDY_CITIES

# The rate study's primary holdout is also seed 17. This draw is a panel of
# listings, not that holdout. A later occupancy retest needs a new seed.
SAMPLE_SEED = 17
TARGET_CALENDARS = 300
# A larger ask than this does not fit a two-hour city job at the 2 second delay.
MAX_PANEL = 1200
SAMPLE_CITIES: tuple[str, ...] = STUDY_CITIES
REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "data" / "calendar_sample.json"
POOL_DIR = REPO_ROOT / "data" / "calendar_pool"
ACTIVE_RULE = "number_of_reviews_ltm >= 1 and availability_365 >= 1"


def listing_id(value: object) -> str | None:
    """One decimal form of a listing id, without rounding through float.

    Newer Airbnb ids are longer than the 53-bit float mantissa. Reading one
    as a float and converting back changes the id, and the public calendar
    then comes back empty. Integers and digit strings stay exact. A float
    that large has already lost digits, so it is rejected.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if np.isnan(value) or abs(value) >= 2**53:
            return None
        if value.is_integer():
            return str(int(value))
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    if text.lstrip("-").isdigit():
        return str(int(text))
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _stratum_label(neighbourhood: object, room_type: object) -> str:
    hood = str(neighbourhood).strip() if pd.notna(neighbourhood) else ""
    room = str(room_type).strip() if pd.notna(room_type) else ""
    return f"{hood or 'unknown'}|{room or 'unknown'}"


def active_mask(frame: pd.DataFrame) -> pd.Series:
    """Homes with a review in the file's last twelve months and an open night.

    ``number_of_reviews_ltm`` is Inside Airbnb's count of reviews in the twelve
    months before the scrape. ``availability_365`` is open nights in the next
    year on that same scrape. Both have to be at least 1.
    """
    reviews = pd.to_numeric(frame["number_of_reviews_ltm"], errors="coerce").fillna(0)
    available = pd.to_numeric(frame["availability_365"], errors="coerce").fillna(0)
    return (reviews >= 1) & (available >= 1)


def panel_size(
    returned: int,
    probed: int,
    target: int = TARGET_CALENDARS,
    cap: int = MAX_PANEL,
) -> int:
    """How many ids to ask so about ``target`` calendars come back.

    ``returned / probed`` is the share of a probe that published a calendar.
    The result is rounded up. It stops at ``cap`` when the probe return rate
    is too low for one city job. A probe that returned nothing asks ``cap``.
    """
    if probed <= 0:
        raise ValueError("probe asked nothing")
    if returned < 0 or returned > probed:
        raise ValueError("returned count is outside the probe")
    if returned == 0:
        return cap
    asked = math.ceil(target * probed / returned)
    return min(cap, asked)


def load_strata(city_key: str, path: Path | None = None) -> pd.DataFrame:
    """One row per eligible listing id, with the stratum used for the draw.

    The default file is the committed eligible pool, already filtered.
    A raw listings file can be passed instead; rows that fail ``active_mask``
    are dropped when that file still has the review and availability columns.
    """
    source = path or (POOL_DIR / f"{city_key}.csv.gz")
    header = pd.read_csv(source, nrows=0).columns
    id_dtype = {column: "string" for column in ("id", "listing_id") if column in header}
    frame = pd.read_csv(source, dtype=id_dtype)
    if "listing_id" not in frame.columns:
        frame["listing_id"] = frame["id"].map(listing_id)
    else:
        frame["listing_id"] = frame["listing_id"].map(listing_id)
    frame = frame[frame["listing_id"].notna()].drop_duplicates("listing_id")
    if {"number_of_reviews_ltm", "availability_365"}.issubset(frame.columns):
        frame = frame[active_mask(frame)]
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


def _sizes(per_city: int | Mapping[str, int], cities: Sequence[str]) -> dict[str, int]:
    if isinstance(per_city, int):
        return {city: per_city for city in cities}
    missing = [city for city in cities if city not in per_city]
    if missing:
        raise ValueError(f"No panel size for: {', '.join(missing)}")
    return {city: int(per_city[city]) for city in cities}


def build_sample_document(
    cities: Sequence[str] = SAMPLE_CITIES,
    per_city: int | Mapping[str, int] = TARGET_CALENDARS,
    seed: int = SAMPLE_SEED,
    extra: Mapping[str, object] | None = None,
) -> dict:
    """The document committed at ``data/calendar_sample.json``.

    ``per_city`` is how many ids to ask, not how many calendars come back.
    The probe that chose those counts belongs in ``extra``.
    """
    sizes = _sizes(per_city, cities)
    payload: dict = {
        "seed": seed,
        "target_calendars": TARGET_CALENDARS,
        "active": ACTIVE_RULE,
        "strata": ["neighbourhood_cleansed", "room_type"],
        "allocation": "largest remainder, capped at the stratum size",
        "draw": "SeedSequence([seed, city_index]); pools sorted; ids sorted",
        "cities": {},
    }
    if extra:
        payload.update(dict(extra))
    for index, city in enumerate(cities):
        frame = load_strata(city)
        ids = draw_ids(frame, sizes[city], seed, index)
        payload["cities"][city] = {
            "eligible": int(len(frame)),
            "strata": int(frame["stratum"].nunique()),
            "asked": sizes[city],
            "sampled": len(ids),
            "listing_ids": ids,
        }
    return payload


def write_sample(path: Path | None = None) -> dict:
    """Rewrite the panel from the pools, keeping the committed ask sizes."""
    dest = path or SAMPLE_PATH
    current = json.loads(dest.read_text(encoding="utf-8"))
    sizes = {
        city: int(body.get("asked", body["sampled"])) for city, body in current["cities"].items()
    }
    extra = {key: current[key] for key in ("sources", "probe", "note") if key in current}
    document = build_sample_document(per_city=sizes, seed=int(current["seed"]), extra=extra)
    for city, body in document["cities"].items():
        previous = current["cities"].get(city, {})
        for key, value in previous.items():
            if key not in body:
                body[key] = value
    dest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def load_sample(path: Path | None = None) -> dict[str, list[str]]:
    """City key to the committed listing ids, as strings."""
    document = json.loads((path or SAMPLE_PATH).read_text(encoding="utf-8"))
    return {
        city: [str(listing_id) for listing_id in body["listing_ids"]]
        for city, body in document["cities"].items()
    }
