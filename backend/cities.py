"""Shim so the local app can import city metadata from data.cities."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.cities import (  # noqa: E402
    AIRPORTS,
    CITIES,
    DEFAULT_CITY,
    SEASONAL_FACTORS,
    STUDY_CITIES,
    airport_points,
    get_city,
    get_city_or_default,
    get_seasonal_factors,
    list_cities,
)

__all__ = [
    "AIRPORTS",
    "CITIES",
    "DEFAULT_CITY",
    "SEASONAL_FACTORS",
    "STUDY_CITIES",
    "airport_points",
    "get_city",
    "get_city_or_default",
    "get_seasonal_factors",
    "list_cities",
]
