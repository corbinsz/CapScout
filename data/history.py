"""Earlier Inside Airbnb snapshots, as extra training rows.

The study file stays the population we score. A snapshot dated after that
file would be a price from the future, so it is not loaded. Listings that
are already in the study file are not repeated: the newest earlier snapshot
is kept only for ids the study file does not have. Host tenure is measured
as of that snapshot, not as of the study date.

`python -m data.history` downloads anything missing. The evaluation calls
the same function.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from evaluation.protocol import HISTORY_SNAPSHOTS
from features.place import attach_place_features

from data.cities import CITIES
from data.ingest import RAW_ROOT, load_listings

_UA = "CapScout personal research (CC BY 4.0 Inside Airbnb snapshot)"
_IA = "https://data.insideairbnb.com/united-states"


def history_dir(city_key: str) -> Path:
    path = RAW_ROOT / city_key / "history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def history_path(city_key: str, snapshot: str) -> Path:
    return history_dir(city_key) / f"{snapshot}.csv.gz"


def _months_before(study_date: str, snapshot: str) -> float:
    study = pd.Timestamp(study_date)
    earlier = pd.Timestamp(snapshot)
    return float((study - earlier).days / 30.4375)


def _retarget_tenure(frame: pd.DataFrame, as_of: str) -> None:
    """Host tenure as of the snapshot the row came from, not the study date."""
    if "host_since" not in frame.columns:
        frame["log_host_tenure"] = np.nan
        return
    since = pd.to_datetime(frame["host_since"], errors="coerce")
    years = (pd.Timestamp(as_of) - since).dt.days / 365.25
    years = years.where(years >= 0)
    frame["log_host_tenure"] = np.log1p(years.to_numpy(dtype=float))


def listing_id_set(series: pd.Series) -> set[str]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype("int64")
    return {str(value) for value in values.tolist()}


def dedupe_snapshots(frames: list[pd.DataFrame], study_ids: set[str]) -> pd.DataFrame:
    """Newest snapshot wins. Ids already in the study file are dropped.

    `frames` must already be in newest-first order. An empty input is an
    empty frame, which the caller treats as "no extra rows".
    """
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)
    stacked["_id"] = pd.to_numeric(stacked["id"], errors="coerce")
    stacked = stacked[stacked["_id"].notna()].copy()
    stacked["_id"] = stacked["_id"].astype("int64").astype(str)
    stacked = stacked.drop_duplicates("_id", keep="first")
    stacked = stacked[~stacked["_id"].isin(study_ids)].copy()
    stacked["id"] = stacked["_id"].astype("int64")
    return stacked.drop(columns=["_id"]).reset_index(drop=True)


def download_history(cities: tuple[str, ...] | None = None) -> None:
    """Download the pre-registered snapshots that are not already on disk."""
    cities = tuple(cities) if cities is not None else tuple(HISTORY_SNAPSHOTS)
    for city in cities:
        state = CITIES[city]["ia_state"]
        slug = CITIES[city]["ia_city"]
        for snapshot in HISTORY_SNAPSHOTS[city]:
            dest = history_path(city, snapshot)
            if dest.exists() and dest.stat().st_size > 1_000_000:
                continue
            url = f"{_IA}/{state}/{slug}/{snapshot}/data/listings.csv.gz"
            print(f"Downloading {city} {snapshot}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=180) as resp:
                dest.write_bytes(resp.read())


def load_history(city_key: str, study_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Cleaned extra rows for one city, with census and map columns attached.

    The study date is the zero of `months_before`. Every row here is positive.
    """
    study_date = CITIES[city_key]["ia_date"]
    # Newest first, so dedupe keeps the snapshot closest to the study date.
    dates = tuple(sorted(HISTORY_SNAPSHOTS[city_key], reverse=True))
    frames = []
    for snapshot in dates:
        path = history_path(city_key, snapshot)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Download it with: python -m data.history"
            )
        frame, stats = load_listings(city_key, path=path, snapshot=snapshot)
        _retarget_tenure(frame, snapshot)
        frame["months_before"] = _months_before(study_date, snapshot)
        frame["history_snapshot"] = snapshot
        frames.append(frame)
        print(
            f"  history {city_key} {snapshot}: {stats['rows_final']} rows after cleaning",
            flush=True,
        )
    extra = dedupe_snapshots(frames, study_ids)
    if len(extra):
        extra, _info = attach_place_features(extra, city_key)
    counts = {
        "snapshots": list(HISTORY_SNAPSHOTS[city_key]),
        "rows_before_dedupe": int(sum(len(frame) for frame in frames)),
        "rows_after_dedupe": int(len(extra)),
    }
    print(
        f"  history {city_key}: {counts['rows_after_dedupe']} listings not in the study file",
        flush=True,
    )
    return extra, counts


def ensure_history(cities: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Download, clean, and dedupe. Keyed by city. The study frame is separate."""
    download_history(cities)
    loaded = {}
    for city in cities:
        study, _stats = load_listings(city)
        extra, _counts = load_history(city, listing_id_set(study["id"]))
        loaded[city] = extra
    return loaded


def main(argv: list[str] | None = None) -> None:
    args = [a for a in (argv if argv is not None else sys.argv[1:]) if not a.startswith("-")]
    cities = tuple(args) if args else tuple(HISTORY_SNAPSHOTS)
    ensure_history(cities)


if __name__ == "__main__":
    main()
