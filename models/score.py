"""Score one property with the saved development-set model.

Listing-quality fields (reviews, amenities, host, minimum nights) are left
missing unless the caller actually has them. Filling them with zeros would
claim the home has no kitchen and no reviews. Trees treat missingness as
its own direction, which is the honest input for a house that has never
been listed. The published test metrics are for listings where those fields
were observed; a for-sale score is a rougher use of the same model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from evaluation.fitting import predict_dollars
from features.frame import AMENITY_FLAGS, attach_base_features

from models.artifacts import available, load_artifact


def model_available(city_key: str) -> bool:
    return available(city_key)


def _frame_row(features: dict, city_key: str) -> pd.Series:
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
    # attach_base_features treats a missing review count as "zero reviews",
    # which is right for a listing file and wrong for a home with no ad.
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
    return frame.iloc[0]


def score_features(features: dict, city_key: str) -> dict:
    artifact = load_artifact(city_key)
    row = _frame_row(features, city_key)
    columns = list(artifact["columns"])
    matrix = pd.DataFrame([{column: row.get(column, np.nan) for column in columns}])
    if artifact["use_spatial"]:
        neighbourhood = np.array([features.get("neighbourhood") or "unknown"])
        neigh, bed, knn = artifact["encoder"].transform_new(
            np.array([float(features["latitude"])]),
            np.array([float(features["longitude"])]),
            neighbourhood,
            np.array([float(row["bedrooms"])]),
        )
        matrix["neigh_log_price"] = neigh
        matrix["neigh_bed_log_price"] = bed
        matrix["knn_log_price"] = knn
    nightly = float(predict_dollars(artifact["model"], matrix, artifact["library"])[0])
    neighbourhood_name = features.get("neighbourhood") or ""
    occupancy = artifact["occupancy_by_neighbourhood"].get(
        neighbourhood_name, artifact["occupancy_global"]
    )
    return {
        "nightly_rate": nightly,
        "occupancy_rate": float(occupancy),
        "occupancy_source": "neighbourhood_median_of_review_proxy",
        "feature_set": artifact["feature_set"],
        "library": artifact["library"],
    }
