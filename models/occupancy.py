"""Score booked nights, annual revenue, and the city grade for one home.

The nights model is separate from the nightly-rate artifact. Review counts
are not inputs. A missing amenity stays missing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from data.ingest import load_listings
from evaluation.fitting import predict_log
from features.columns import CENSUS_COLUMNS
from features.frame import AMENITY_FLAGS, EXTRA_AMENITY_FLAGS, attach_base_features
from features.geo import haversine_km
from features.place import features_at

from models.grade import annual_revenue, grade_revenue, gross_yield

REPO = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("Low", "Medium", "High")


def occupancy_path(city_key: str) -> Path:
    return REPO / "models" / "artifacts" / city_key / "occupancy.joblib"


def occupancy_available(city_key: str) -> bool:
    return occupancy_path(city_key).is_file()


def load_occupancy(city_key: str) -> dict:
    path = occupancy_path(city_key)
    if not path.exists():
        raise FileNotFoundError(
            f"No occupancy model for {city_key}. Run: python -m evaluation.run_occupancy"
        )
    return joblib.load(path)


def _predict_nights(
    model, matrix: pd.DataFrame, library: str, logged: bool, cap: float
) -> np.ndarray:
    raw = predict_log(model, matrix, library)
    if logged:
        raw = np.expm1(raw)
    return np.clip(np.maximum(raw, 0.0), 0.0, cap)


def _matrix_prediction(artifact: dict, matrix: pd.DataFrame) -> float:
    cap = float(artifact["annual_cap"])
    logged = bool(artifact["logged"])
    preds = [
        _predict_nights(artifact["models"][library], matrix, library, logged, cap)
        for library in artifact["libraries"]
    ]
    return float(np.mean(np.vstack(preds), axis=0)[0])


def _row(features: dict, city_key: str, needed: set[str]) -> pd.Series:
    bedrooms = float(features.get("bedrooms") or 1)
    row = {
        "bedrooms": bedrooms,
        "bathrooms_clean": features.get("bathrooms", np.nan),
        "accommodates": features.get("accommodates") or bedrooms * 2,
        "latitude": float(features["latitude"]),
        "longitude": float(features["longitude"]),
        "property_type": features.get("property_type") or "",
        "room_type": features.get("room_type") or "Entire home/apt",
        "beds": features.get("beds", np.nan),
        "amenities": features.get("amenities", np.nan),
        "minimum_nights": np.nan,
        "number_of_reviews": np.nan,
        "reviews_per_month": np.nan,
        "neighbourhood_cleansed": "unknown",
    }
    frame = attach_base_features(pd.DataFrame([row]), city_key)
    if "amenities" not in features:
        for column in ("amenity_count", *AMENITY_FLAGS, *EXTRA_AMENITY_FLAGS):
            frame[column] = np.nan
    if "beds" not in features:
        frame["beds"] = np.nan
    if needed & set(CENSUS_COLUMNS):
        place = features_at(city_key, float(features["latitude"]), float(features["longitude"]))
        for column in needed & set(CENSUS_COLUMNS):
            frame[column] = place[column]
    return frame.iloc[0]


@lru_cache(maxsize=8)
def _places(city_key: str) -> pd.DataFrame:
    frame, _stats = load_listings(city_key)
    return frame[["latitude", "longitude", "neighbourhood_cleansed"]].reset_index(drop=True)


def _nearest_neighbourhood(city_key: str, lat: float, lon: float) -> str | None:
    frame = _places(city_key)
    if frame.empty:
        return None
    distance = haversine_km(frame["latitude"].to_numpy(), frame["longitude"].to_numpy(), lat, lon)
    return str(frame.iloc[int(np.argmin(distance))]["neighbourhood_cleansed"])


def _class_probabilities(artifact: dict, matrix: pd.DataFrame) -> dict[str, float] | None:
    model = artifact.get("classifier")
    if model is None:
        return None
    proba = np.asarray(model.predict_proba(matrix), dtype=float)[0]
    index = {int(label): pos for pos, label in enumerate(model.classes_)}
    if any(label not in index for label in (0, 1, 2)):
        return None
    return {name: float(proba[index[label]]) for label, name in enumerate(CLASS_NAMES)}


def predict_stay(
    features: dict,
    city_key: str,
    nightly_rate: float | None = None,
    purchase_price: float | None = None,
) -> dict:
    """Nights, revenue, grade, and the neighbourhood median for one home."""
    artifact = load_occupancy(city_key)
    needed = set(artifact["columns"])
    row = _row(features, city_key, needed)
    matrix = pd.DataFrame([row])[list(artifact["columns"])].apply(pd.to_numeric, errors="coerce")
    nights = _matrix_prediction(artifact, matrix)
    neighbourhood = _nearest_neighbourhood(
        city_key, float(features["latitude"]), float(features["longitude"])
    )
    area = artifact["area_medians"].get(neighbourhood, artifact["city_median_nights"])
    if neighbourhood not in artifact["area_medians"]:
        neighbourhood = None
    out = {
        "booked_nights": nights,
        "area_median_nights": float(area),
        "area_name": neighbourhood,
        "annual_revenue": None,
        "grade": None,
        "grade_words": None,
        "gross_yield": None,
        "occupancy_class": None,
        "class_probability": None,
        "class_probabilities": None,
    }
    cuts = artifact.get("class_cuts") or {}
    low_max = cuts.get("low_max")
    high_min = cuts.get("high_min")
    if low_max is not None and high_min is not None:
        if nights <= float(low_max):
            out["occupancy_class"] = "Low"
        elif nights > float(high_min):
            out["occupancy_class"] = "High"
        else:
            out["occupancy_class"] = "Medium"
    probabilities = _class_probabilities(artifact, matrix)
    if probabilities:
        out["class_probabilities"] = probabilities
        named = max(probabilities, key=probabilities.get)
        out["class_probability"] = probabilities[named]
    if nightly_rate is not None:
        revenue = annual_revenue(nightly_rate, nights)
        graded = grade_revenue(revenue, artifact["grade_cuts"])
        out["annual_revenue"] = revenue
        out["grade"] = graded["letter"]
        out["grade_words"] = graded["words"]
        out["gross_yield"] = gross_yield(revenue, purchase_price)
    return out
