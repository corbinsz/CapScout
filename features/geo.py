"""Distances that can be computed for any coordinate, with no price data."""

from __future__ import annotations

import numpy as np
from data.cities import airport_points, get_city_or_default


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Great-circle distance in kilometers. lat2/lon2 are a single point."""
    radius = 6371.0
    lat1_r = np.radians(np.asarray(lat1, dtype=float))
    lon1_r = np.radians(np.asarray(lon1, dtype=float))
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_to_center_km(lat: np.ndarray, lon: np.ndarray, city_key: str) -> np.ndarray:
    city = get_city_or_default(city_key)
    return haversine_km(lat, lon, city["latitude"], city["longitude"])


def distance_to_airport_km(lat: np.ndarray, lon: np.ndarray, city_key: str) -> np.ndarray:
    """Kilometers to the nearest FAA reference point configured for the city."""
    points = airport_points(city_key)
    stacked = [haversine_km(lat, lon, plat, plon) for plat, plon in points]
    return np.min(np.vstack(stacked), axis=0)


def distance_to_city_center_km(lat: np.ndarray, lon: np.ndarray, city_key: str) -> np.ndarray:
    """Kilometers from the city's reference point. Same inputs as the downtown feature."""
    return distance_to_center_km(lat, lon, city_key)
