"""Score one home with the shipped simple-study artifact.

Blank fields stay missing. The model's imputer fills those with the training
median, which is the median training home, not a claim that the field is zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from data.clean import load_clean_city
from evaluation.study_models import apply_interval
from features.columns import pretty
from features.frame import _amenity_frame
from features.geo import distance_to_airport_km, distance_to_center_km
from features.places import attraction_index, distance_to_coast_km

from models.artifacts import load_artifact
from models.grade import annual_revenue, grade_revenue, gross_yield

HOUSE_TYPES = {"House", "Townhouse", "Cabin", "Guest unit", "Cottage"}
ROOM_FLAGS = ("is_private_room", "is_shared_room", "is_hotel_room")
FLAG_COLUMNS = {"bath_is_shared", "is_superhost"}


def nearby_comps(
    frame: pd.DataFrame,
    lat: float,
    lon: float,
    room: str,
    bedrooms: float,
    n: int = 5,
) -> pd.DataFrame:
    """Nearest advertised listings. Same room type, bedrooms within 1. Not model inputs."""
    if frame is None or len(frame) == 0:
        return frame.iloc[0:0].copy() if frame is not None else pd.DataFrame()
    work = frame.copy()
    work["km"] = _km(work["latitude"].to_numpy(), work["longitude"].to_numpy(), lat, lon)
    same = work[(work["room_type"] == room) & (work["bedrooms"].sub(bedrooms).abs() <= 1)]
    pool = same if len(same) >= n else work[work["room_type"] == room]
    if len(pool) < n:
        pool = work
    return pool.nsmallest(n, "km").reset_index(drop=True)


def load_comp_frame(city_key: str) -> pd.DataFrame:
    """Cleaned listings for the map. Empty if the snapshot has not been downloaded."""
    columns = [
        "latitude",
        "longitude",
        "price_clean",
        "bedrooms",
        "bathrooms",
        "room_type",
        "neighbourhood_cleansed",
    ]
    try:
        frame, _report = load_clean_city(city_key)
    except FileNotFoundError:
        return pd.DataFrame(columns=columns)
    return frame[columns].reset_index(drop=True)


def score_home(city_key: str, spec: dict, artifact: dict | None = None) -> dict:
    """Nightly rate, range, nights, revenue, grade, and plain reasons."""
    artifact = load_artifact(city_key) if artifact is None else artifact
    if artifact.get("kind") != "simple-study":
        raise RuntimeError(f"{city_key} artifact is not the simple study model")
    price_model = artifact["model"]
    nights_model = artifact["nights_model"]
    row, blanks = _feature_row(city_key, spec, price_model.columns, nights_model.columns)
    rate = float(price_model.predict(row)[0])
    low, high = apply_interval(np.array([rate]), artifact["interval"])
    nights = float(nights_model.predict(row)[0])
    revenue = annual_revenue(rate, nights)
    grade = grade_revenue(revenue, artifact["grade_cuts"])
    neighbourhood = spec.get("neighbourhood")
    area_name, area_nights = _area_nights(artifact, neighbourhood)
    reasons = _reasons(price_model, row)
    return {
        "nightly_rate": rate,
        "interval_low": float(low[0]),
        "interval_high": float(high[0]),
        "interval_coverage": float(artifact["interval_coverage"]),
        "interval_median_width": float(artifact["interval_median_width"]),
        "booked_nights": nights,
        "nights_rough_estimate": bool(artifact["nights_rough_estimate"]),
        "nights_holdout_mae": float(artifact["nights_holdout_mae"]),
        "nights_holdout_r2": float(artifact["nights_holdout_r2"]),
        "nights_baseline_mae": float(artifact["nights_baseline_mae"]),
        "annual_revenue": revenue,
        "grade": grade["letter"],
        "grade_words": grade["words"],
        "gross_yield": gross_yield(revenue, spec.get("purchase_price")),
        "reasons": reasons,
        "library": artifact["library"],
        "nights_library": artifact["nights_library"],
        "blank_fields": blanks,
        "area_name": area_name,
        "area_median_nights": area_nights,
        "price_holdout_mae": float(artifact["price_holdout_mae"]),
        "price_holdout_r2": float(artifact["price_holdout_r2"]),
    }


def _feature_row(city_key: str, spec: dict, price_columns: list[str], nights_columns: list[str]):
    columns = list(dict.fromkeys([*price_columns, *nights_columns]))
    values = {column: np.nan for column in columns}
    lat = float(spec["latitude"])
    lon = float(spec["longitude"])
    _put(values, "latitude", lat)
    _put(values, "longitude", lon)
    _put(values, "distance_to_center_km", _one(distance_to_center_km(lat, lon, city_key)))
    _put(values, "distance_to_airport_km", _one(distance_to_airport_km(lat, lon, city_key)))
    _put(values, "attraction_index", _one(attraction_index([lat], [lon], city_key)))
    if "distance_to_coast_km" in values:
        _put(values, "distance_to_coast_km", _one(distance_to_coast_km([lat], [lon], city_key)))
    for column in ("bedrooms", "bathrooms", "accommodates", "beds"):
        if spec.get(column) is not None:
            _put(values, column, float(spec[column]))
    _room_flags(values, spec.get("room_type"))
    _building_flags(values, spec.get("property_type"))
    _amenity_flags(values, spec.get("amenities"))
    for column, value in (spec.get("amenity_flags") or {}).items():
        _put(values, column, float(value))
    if spec.get("review_score") is not None:
        _put(values, "review_score", float(spec["review_score"]))
    if spec.get("number_of_reviews") is not None:
        _put(values, "log_reviews", float(np.log1p(float(spec["number_of_reviews"]))))
    if spec.get("host_is_superhost") is not None:
        _put(values, "is_superhost", _yes(spec["host_is_superhost"]))
    if spec.get("bath_is_shared") is not None:
        _put(values, "bath_is_shared", _yes(spec["bath_is_shared"]))
    bedrooms = values.get("bedrooms")
    guests = values.get("accommodates")
    if (
        "bedrooms_x_accommodates" in values
        and bedrooms is not None
        and guests is not None
        and np.isfinite(bedrooms)
        and np.isfinite(guests)
    ):
        _put(values, "bedrooms_x_accommodates", float(bedrooms) * float(guests))
    if any(column.startswith("census_") for column in columns):
        _put_census(values, city_key, lat, lon)
    row = pd.DataFrame([values])
    blanks = [column for column in price_columns if not np.isfinite(row.iloc[0][column])]
    return row, blanks


def _one(value) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _put(values: dict, column: str, value: float) -> None:
    if column in values:
        values[column] = value


def _room_flags(values: dict, room: str | None) -> None:
    if room is None or not any(column in values for column in ROOM_FLAGS):
        return
    for column in ROOM_FLAGS:
        _put(values, column, 0.0)
    if room == "Private room":
        _put(values, "is_private_room", 1.0)
    elif room == "Shared room":
        _put(values, "is_shared_room", 1.0)
    elif room == "Hotel room":
        _put(values, "is_hotel_room", 1.0)


def _building_flags(values: dict, property_type: str | None) -> None:
    if property_type in HOUSE_TYPES:
        _put(values, "is_house", 1.0)
        _put(values, "is_condo", 0.0)
    elif property_type == "Condo":
        _put(values, "is_house", 0.0)
        _put(values, "is_condo", 1.0)


def _amenity_flags(values: dict, amenities: str | None) -> None:
    if amenities is None:
        return
    frame = _amenity_frame(pd.Series([amenities]))
    for column in frame.columns:
        _put(values, column, float(frame.iloc[0][column]))


def _put_census(values: dict, city_key: str, lat: float, lon: float) -> None:
    try:
        from evaluation.study_data import census_at

        found = census_at(city_key, lat, lon)
    except Exception:
        return
    for column, value in found.items():
        _put(values, column, value)


def _yes(value) -> float:
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"t", "true", "1", "yes", "y"} else 0.0
    return 1.0 if value else 0.0


def _area_nights(artifact: dict, neighbourhood) -> tuple[str | None, float]:
    lookup = artifact.get("nights_by_neighbourhood") or {}
    name = None if neighbourhood is None else str(neighbourhood)
    if name and name in lookup:
        return name, float(lookup[name])
    return None, float(artifact["nights_city_median"])


def _reasons(model, row: pd.DataFrame) -> list[dict]:
    """Dollar change from putting one filled-in feature back to its training median."""
    base = float(model.predict(row)[0])
    reasons = []
    for column in model.columns:
        value = row.iloc[0][column]
        if not np.isfinite(value):
            continue
        median = float(model.medians_[column])
        if abs(float(value) - median) < 1e-9:
            continue
        altered = row.copy()
        altered[column] = median
        delta = base - float(model.predict(altered)[0])
        if abs(delta) < 1.0:
            continue
        reasons.append(
            {
                "feature": column,
                "label": pretty(column),
                "dollars": delta,
                "sentence": _sentence(column, float(value), median, delta),
            }
        )
    reasons.sort(key=lambda item: abs(item["dollars"]), reverse=True)
    return reasons[:5]


def _sentence(column: str, value: float, median: float, delta: float) -> str:
    name = pretty(column)
    if column == "distance_to_center_km":
        fact = (
            f"{name} is {_num(value)} km. "
            f"The median training home is about {_num(median)} km from downtown"
        )
    elif column == "attraction_index":
        fact = (
            f"The public-place score is {_num(value)}. "
            f"The median training home's score is {_num(median)}"
        )
    elif column.startswith(("is_", "has_")) or column in FLAG_COLUMNS:
        now = "yes" if value >= 0.5 else "no"
        usual = "yes" if median >= 0.5 else "no"
        fact = f"{name} is {now}. The median training home is {usual}"
    else:
        fact = f"{name} is {_num(value)}. The median training home is {_num(median)}"
    amount = f"${abs(delta):,.0f}"
    if delta > 0:
        move = f"Putting it back to that median lowers the nightly rate by {amount}."
    else:
        move = f"Putting it back to that median raises the nightly rate by {amount}."
    return f"{fact}. {move}"


def _num(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def _km(lat, lon, lat2, lon2) -> np.ndarray:
    from features.geo import haversine_km

    return haversine_km(lat, lon, lat2, lon2)
