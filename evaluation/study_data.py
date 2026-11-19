"""Modeling frame: cleaned listings, distances, and census tracts.

Census columns are a function of the coordinate. They are attached before
the split because they do not use the nightly rate. OpenStreetMap is not
attached. An earlier round already measured it, and this round keeps the
feature list to things a person can name.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from data.clean import load_clean_city
from features.columns import CENSUS_COLUMNS
from features.frame import attach_base_features
from features.place import EXTERNAL, PlaceIndex, _as_polygons
from shapely import STRtree

from evaluation.study_split import cell_ids


def repair_superhost(frame: pd.DataFrame) -> pd.DataFrame:
    """Unknown superhost status stays missing. It is not a 'no'."""
    if "host_is_superhost" not in frame.columns:
        frame["is_superhost"] = np.nan
        return frame
    text = frame["host_is_superhost"].astype(str).str.lower().str.strip()
    known = text.isin(["t", "true", "1", "f", "false", "0"]).to_numpy()
    values = np.full(len(frame), np.nan)
    values[known] = text.isin(["t", "true", "1"]).to_numpy()[known].astype(float)
    frame["is_superhost"] = values
    return frame


def collapse_building(frame: pd.DataFrame) -> pd.DataFrame:
    """House, townhouse, cabin, cottage, and guest unit are 'a house'.

    Condo and apartment stay the other flag. Hotel rooms are a room type,
    not a building flag.
    """
    parts = np.vstack(
        [
            frame[column].to_numpy(dtype=float)
            for column in ("is_house", "is_townhouse", "is_cabin", "is_cottage", "is_guest")
        ]
    )
    frame["is_house"] = np.nanmax(parts, axis=0).astype(np.int8)
    return frame


@lru_cache(maxsize=4)
def census_index(city: str) -> PlaceIndex:
    directory = EXTERNAL / city
    payload = json.loads((directory / "tracts.geojson").read_text())
    meta = json.loads((directory / "context.json").read_text())
    geoms, properties = _as_polygons(payload["features"])
    if not geoms:
        raise RuntimeError(f"{city} tract file has no polygons")
    centroids = np.array([geom.centroid.coords[0] for geom in geoms], dtype=float)
    medians = {name: float(meta["census_medians"][name]) for name in CENSUS_COLUMNS}
    bbox = tuple(float(value) for value in meta["bbox"])
    return PlaceIndex(
        city=city,
        geoms=geoms,
        tree=STRtree(geoms),
        properties=properties,
        centroids_lon=centroids[:, 0],
        centroids_lat=centroids[:, 1],
        medians=medians,
        pois={},
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
    )


def census_at(city: str, lat: float, lon: float) -> dict[str, float]:
    row = census_index(city).census_for(np.array([lat]), np.array([lon])).iloc[0]
    return {column: float(row[column]) for column in CENSUS_COLUMNS}


def prepare_city(city_key: str) -> tuple[pd.DataFrame, dict]:
    frame, report = load_clean_city(city_key)
    frame = attach_base_features(frame, city_key)
    frame = repair_superhost(frame)
    frame = collapse_building(frame)
    census = census_index(city_key).census_for(
        frame["latitude"].to_numpy(dtype=float),
        frame["longitude"].to_numpy(dtype=float),
    )
    for column in CENSUS_COLUMNS:
        frame[column] = census[column].to_numpy()
    join = census["census_join"].to_numpy()
    report["census"] = {
        "vintage": "ACS 2024 5-year",
        "tables": "B19013 income, B25064 rent, B25077 home value, B01003 population",
        "boundaries": "TIGER/Line 2024 census tracts",
        "license": "U.S. government work, public domain",
        "polygon_fraction": float(np.mean(join == 0)),
        "centroid_fallback_fraction": float(np.mean(join == 1)),
        "city_median_fraction": float(np.mean(join == 2)),
        "field_imputed_fraction": float(census["census_imputed"].mean()),
    }
    frame["cell"] = cell_ids(
        frame["latitude"].to_numpy(dtype=float),
        frame["longitude"].to_numpy(dtype=float),
        city_key,
    )
    frame = frame.drop(columns=["number_of_reviews_ltm", "occupancy_rate"], errors="ignore")
    frame = frame.reset_index(drop=True)
    if not np.isfinite(frame["price_clean"].to_numpy()).all():
        raise RuntimeError(f"{city_key} has a non-finite price after cleaning")
    if not np.isfinite(frame["booked_nights"].to_numpy()).all():
        raise RuntimeError(f"{city_key} has a non-finite nights label after cleaning")
    report["rows_model"] = int(len(frame))
    report["n_cells"] = int(frame["cell"].nunique())
    return frame, report
