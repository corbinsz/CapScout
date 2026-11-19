"""Fixed public places used by the pre-registered feature try.

The lists were written down before the training-cell scores were read.
They are not tuned on the holdout. Distance to the coast is the distance
to the nearest point on a short public shoreline list. Seattle's list is
the Sound, not the open Pacific. Austin has no coast, so that column is
not created there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.geo import haversine_km

# Inverse-distance floor, in kilometers, so a listing on top of a landmark
# does not send the index to infinity.
_FLOOR_KM = 0.25

ATTRACTIONS: dict[str, tuple[tuple[float, float], ...]] = {
    "austin": (
        (30.2747, -97.7404),  # Texas State Capitol
        (30.2641, -97.7713),  # Barton Springs
        (30.2862, -97.7394),  # University of Texas Tower
        (30.2616, -97.7453),  # Congress Avenue bridge
        (30.2670, -97.7729),  # Zilker Park
        (30.2656, -97.7518),  # Austin Central Library
    ),
    "san-diego": (
        (32.7313, -117.1447),  # Balboa Park
        (32.7138, -117.1752),  # USS Midway
        (32.8505, -117.2730),  # La Jolla Cove
        (32.7543, -117.1964),  # Old Town
        (32.6804, -117.1776),  # Coronado Beach
        (32.6735, -117.2425),  # Cabrillo National Monument
    ),
    "seattle": (
        (47.6097, -122.3422),  # Pike Place Market
        (47.6205, -122.3493),  # Space Needle
        (47.6553, -122.3035),  # University of Washington
        (47.6654, -122.3972),  # Ballard Locks
        (47.6026, -122.3388),  # Colman Dock
        (47.5790, -122.4075),  # Alki Beach
    ),
}

SHORE: dict[str, tuple[tuple[float, float], ...]] = {
    "san-diego": (
        (32.534, -117.123),
        (32.576, -117.132),
        (32.627, -117.143),
        (32.680, -117.178),
        (32.715, -117.232),
        (32.745, -117.253),
        (32.770, -117.253),
        (32.807, -117.262),
        (32.850, -117.273),
        (32.910, -117.255),
        (32.959, -117.265),
        (33.037, -117.292),
        (33.128, -117.336),
        (33.194, -117.384),
    ),
    "seattle": (
        (47.520, -122.360),
        (47.545, -122.395),
        (47.570, -122.410),
        (47.579, -122.408),
        (47.590, -122.380),
        (47.600, -122.338),
        (47.606, -122.355),
        (47.620, -122.380),
        (47.640, -122.395),
        (47.662, -122.410),
        (47.690, -122.405),
        (47.710, -122.395),
        (47.750, -122.370),
    ),
}


def added_columns(city_key: str) -> list[str]:
    """Columns the adopted bundle adds. The coast is omitted where there is none."""
    columns = ["attraction_index"]
    if city_key in SHORE:
        columns.append("distance_to_coast_km")
    columns.append("bedrooms_x_accommodates")
    return columns


def attraction_index(lat, lon, city_key: str) -> np.ndarray:
    """Sum of 1 / distance to each landmark, with distance floored at 0.25 km."""
    places = ATTRACTIONS[city_key]
    total = np.zeros(len(np.asarray(lat, dtype=float)), dtype=float)
    for plat, plon in places:
        distance = haversine_km(lat, lon, plat, plon)
        total = total + 1.0 / np.maximum(distance, _FLOOR_KM)
    return total


def distance_to_coast_km(lat, lon, city_key: str) -> np.ndarray:
    """Kilometers to the nearest listed shoreline point. Only defined for coastal cities."""
    places = SHORE[city_key]
    stacked = [haversine_km(lat, lon, plat, plon) for plat, plon in places]
    return np.min(np.vstack(stacked), axis=0)


def attach_place_features(frame: pd.DataFrame, city_key: str) -> pd.DataFrame:
    """Add the pre-registered columns. Existing columns are left as they are."""
    if city_key not in ATTRACTIONS:
        return frame
    out = frame
    lat = out["latitude"].to_numpy(dtype=float)
    lon = out["longitude"].to_numpy(dtype=float)
    out["attraction_index"] = attraction_index(lat, lon, city_key)
    if city_key in SHORE:
        out["distance_to_coast_km"] = distance_to_coast_km(lat, lon, city_key)
    bedrooms = pd.to_numeric(out["bedrooms"], errors="coerce").to_numpy(dtype=float)
    accommodates = pd.to_numeric(out["accommodates"], errors="coerce").to_numpy(dtype=float)
    out["bedrooms_x_accommodates"] = bedrooms * accommodates
    return out
