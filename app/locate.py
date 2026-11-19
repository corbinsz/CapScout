"""Address lookup for the local app. No network unless the caller passes one.

The Census geocoder is a public-domain address service. A failure, an empty
box, a miss, or a pin more than 60 km from downtown becomes a short message.
The model is not asked to score those cases.
"""

from __future__ import annotations

from data.cities import CITIES
from evaluation.study_protocol import MAX_KM_FROM_CENTER
from features.geo import distance_to_city_center_km

CENSUS_GEOCODER = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"


class LocateError(Exception):
    """A pin or an address the app should show, not score."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def check_pin(lat: float, lon: float, city_key: str) -> None:
    """Reject a coordinate outside the city's 60 km study area."""
    try:
        latitude = float(lat)
        longitude = float(lon)
    except (TypeError, ValueError) as exc:
        raise LocateError("Latitude and longitude need to be numbers.") from exc
    if latitude != latitude or longitude != longitude:
        raise LocateError("Latitude and longitude need to be numbers.")
    distance = float(distance_to_city_center_km(latitude, longitude, city_key))
    if distance > MAX_KM_FROM_CENTER:
        raise LocateError(
            "That point is more than 60 km from downtown, so it is outside "
            "the area this model was built for."
        )


def parse_geocode_response(payload: dict) -> tuple[float, float, str] | None:
    """Read one Census geocoder JSON body. None when the service found nothing."""
    matches = (payload or {}).get("result", {}).get("addressMatches") or []
    if not matches:
        return None
    coords = matches[0].get("coordinates") or {}
    try:
        lat = float(coords["y"])
        lon = float(coords["x"])
    except (KeyError, TypeError, ValueError):
        return None
    label = str(matches[0].get("matchedAddress") or "")
    return lat, lon, label


def locate_address(address: str, city_key: str, fetch) -> dict:
    """Turn a street address into a coordinate inside the study area.

    ``fetch(url, params)`` returns the JSON body or raises. Tests pass a fake.
    """
    text = (address or "").strip()
    if not text:
        raise LocateError("Type an address, or enter a latitude and longitude.")
    city = CITIES[city_key]
    params = {
        "address": f"{text}, {city['name']}, {city['state']}",
        "benchmark": "Public_AR_Current",
        "format": "json",
    }
    try:
        payload = fetch(CENSUS_GEOCODER, params)
    except Exception as exc:
        raise LocateError(
            "The address lookup did not respond. Enter a latitude and longitude instead."
        ) from exc
    found = parse_geocode_response(payload if isinstance(payload, dict) else {})
    if found is None:
        raise LocateError(
            "No match in that city. Try a fuller street address, or enter a latitude and longitude."
        )
    lat, lon, label = found
    check_pin(lat, lon, city_key)
    return {"latitude": lat, "longitude": lon, "label": label}
