"""Score one property with the saved development-set model.

Listing-quality fields are left missing unless the caller has them. Zeros
would claim the home has no kitchen and no reviews. Census and OpenStreetMap
features are looked up from the coordinate, the same join the training rows
used. The interval is the log-residual conformal radius stored with the
artifact. It was fit on development folds, not on this call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from evaluation.fitting import predict_dollars
from evaluation.intervals import apply_log_interval
from features.columns import CENSUS_COLUMNS, OSM_COLUMNS
from features.frame import AMENITY_FLAGS, attach_base_features
from features.place import features_at

from models.artifacts import available, load_artifact
from models.explain import shap_matrix, top_reasons


def model_available(city_key: str) -> bool:
    return available(city_key)


def _frame_row(features: dict, city_key: str, needed: set[str]) -> pd.Series:
    bedrooms = float(features.get("bedrooms") or 1)
    row = {
        "bedrooms": bedrooms,
        "bathrooms_clean": features.get("bathrooms_clean", features.get("bathrooms", np.nan)),
        "accommodates": features.get("accommodates") or bedrooms * 2,
        "latitude": float(features["latitude"]),
        "longitude": float(features["longitude"]),
        "property_type": features.get("property_type") or "",
        "room_type": features.get("room_type") or "Entire home/apt",
        "minimum_nights": features.get("minimum_nights", np.nan),
        "number_of_reviews": features.get("number_of_reviews", np.nan),
        "reviews_per_month": features.get("reviews_per_month", np.nan),
        "review_scores_rating": features.get(
            "review_score", features.get("review_scores_rating", np.nan)
        ),
        "amenities": features.get("amenities", np.nan),
        "host_is_superhost": features.get("host_is_superhost", np.nan),
        "host_listings_count": features.get("host_listings_count", np.nan),
        "instant_bookable": features.get("instant_bookable", np.nan),
        "neighbourhood_cleansed": features.get("neighbourhood") or "unknown",
    }
    frame = attach_base_features(pd.DataFrame([row]), city_key)
    # A missing review count on a file means "no reviews yet". A home that
    # has never been listed is not that. Leave the fields missing.
    if "number_of_reviews" not in features and "reviews_per_month" not in features:
        frame["log_reviews"] = np.nan
        frame["reviews_per_month"] = np.nan
        frame["review_score"] = np.nan
    if "host_is_superhost" not in features:
        frame["is_superhost"] = np.nan
    if "instant_bookable" not in features:
        frame["instant_bookable"] = np.nan
    if "host_listings_count" not in features:
        frame["log_host_listings"] = np.nan
    if "minimum_nights" not in features:
        frame["minimum_nights_capped"] = np.nan
    if "amenities" not in features:
        for column in list(AMENITY_FLAGS) + ["amenity_count"]:
            frame[column] = np.nan
    place_columns = needed & (set(CENSUS_COLUMNS) | set(OSM_COLUMNS))
    if place_columns:
        place = features_at(city_key, float(features["latitude"]), float(features["longitude"]))
        for column in place_columns:
            frame[column] = place[column]
        frame["census_imputed"] = place["census_imputed"]
        frame["census_join"] = place["census_join"]
        frame["inside_extract"] = place["inside_extract"]
    return frame.iloc[0]


def _predict_matrix(artifact: dict, matrix: pd.DataFrame) -> np.ndarray:
    if artifact.get("library") == "blend":
        libraries = list(artifact["blend_libraries"])
        stacked = [
            predict_dollars(artifact["models"][library], matrix, library) for library in libraries
        ]
        return np.mean(stacked, axis=0)
    return predict_dollars(artifact["model"], matrix, artifact["library"])


def _reasons(artifact: dict, matrix: pd.DataFrame) -> list[dict]:
    names = list(matrix.columns)
    if artifact.get("library") == "blend":
        libraries = list(artifact["blend_libraries"])
        values = np.mean(
            [shap_matrix(artifact["models"][library], matrix, library)[0] for library in libraries],
            axis=0,
        )
    else:
        values = shap_matrix(artifact["model"], matrix, artifact["library"])[0]
    observed = matrix.notna().iloc[0].to_numpy()
    return top_reasons(
        names,
        values,
        k=5,
        observed=observed,
        feature_values=matrix.iloc[0].to_numpy(dtype=float),
    )


def score_features(features: dict, city_key: str) -> dict:
    artifact = load_artifact(city_key)
    row = _frame_row(features, city_key, set(artifact["columns"]))
    columns = list(artifact["columns"])
    matrix = pd.DataFrame([{column: row.get(column, np.nan) for column in columns}])
    mode = artifact.get("spatial_mode") or ("both" if artifact.get("use_spatial") else "none")
    if mode not in (None, "none", False):
        neighbourhood = np.array([features.get("neighbourhood") or "unknown"])
        neigh, bed, knn = artifact["encoder"].transform_new(
            np.array([float(features["latitude"])]),
            np.array([float(features["longitude"])]),
            neighbourhood,
            np.array([float(row["bedrooms"])]),
        )
        if mode in ("smooth", "both"):
            matrix["neigh_log_price"] = neigh
            matrix["neigh_bed_log_price"] = bed
        if mode in ("knn", "both"):
            matrix["knn_log_price"] = knn
    nightly = float(_predict_matrix(artifact, matrix)[0])
    radius = artifact.get("conformal_log_radius")
    if radius is None:
        low = high = None
    else:
        low_arr, high_arr = apply_log_interval(np.array([nightly]), float(radius))
        low, high = float(low_arr[0]), float(high_arr[0])
    neighbourhood_name = features.get("neighbourhood") or ""
    occupancy = artifact["occupancy_by_neighbourhood"].get(
        neighbourhood_name, artifact["occupancy_global"]
    )
    return {
        "nightly_rate": nightly,
        "interval_low": low,
        "interval_high": high,
        "interval_level": artifact.get("interval_level", 0.80),
        "reasons": _reasons(artifact, matrix),
        "census_imputed": bool(row.get("census_imputed", False)),
        "census_join": int(row.get("census_join", 0) or 0),
        "inside_extract": bool(row.get("inside_extract", True)),
        "occupancy_rate": float(occupancy),
        "occupancy_source": "neighbourhood_median_of_review_proxy",
        "feature_set": artifact.get("feature_set"),
        "library": artifact.get("library"),
    }
