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
from evaluation.fitting import predict_dollars, predict_log
from evaluation.intervals import (
    apply_log_interval,
    apply_log_radii,
    apply_mondrian,
    apply_signed_interval,
    cqr_bounds,
    sigma_of,
)
from evaluation.protocol import city_dummies
from evaluation.v4 import apply_correction
from features.columns import CENSUS_COLUMNS, LEXICON_COLUMNS, OSM_COLUMNS
from features.frame import AMENITY_FLAGS, EXTRA_AMENITY_FLAGS, attach_base_features
from features.place import features_at

from models.artifacts import available, load_artifact
from models.explain import collapse_wording, plain_reason, shap_matrix, top_reasons


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
        "beds": features.get("beds", np.nan),
        "bathrooms_text": features.get("bathrooms_text", np.nan),
        "host_since": features.get("host_since", np.nan),
        "host_response_rate": features.get("host_response_rate", np.nan),
        "host_acceptance_rate": features.get("host_acceptance_rate", np.nan),
        "host_response_time": features.get("host_response_time", np.nan),
        "host_identity_verified": features.get("host_identity_verified", np.nan),
    }
    if features.get("listing_text") or features.get("description") or features.get("name"):
        row["name"] = features.get("name") or ""
        row["description"] = features.get("description") or features.get("listing_text") or ""
        row["neighborhood_overview"] = features.get("neighborhood_overview") or ""
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
        for column in list(AMENITY_FLAGS) + list(EXTRA_AMENITY_FLAGS) + ["amenity_count"]:
            frame[column] = np.nan
    # Empty strings are not an ad. Leave the lexicon missing so the model
    # treats the wording as unknown rather than as a blank listing.
    supplied_text = bool(
        features.get("listing_text") or features.get("description") or features.get("name")
    )
    if not supplied_text:
        for column in LEXICON_COLUMNS:
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
        weights = artifact.get("blend_weights")
        if weights:
            total = np.zeros(len(matrix), dtype=float)
            for library, column in zip(libraries, stacked, strict=True):
                total = total + float(weights[library]) * column
            return total
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
    feature_values = matrix.iloc[0].to_numpy(dtype=float)
    names, values, observed, feature_values = collapse_wording(
        names, values, observed, feature_values
    )
    reasons = top_reasons(names, values, k=5, observed=observed, feature_values=feature_values)
    for reason in reasons:
        reason["sentence"] = plain_reason(
            reason["feature"], reason.get("value"), reason["approx_percent"]
        )
    return reasons


def _interval(
    artifact: dict, matrix: pd.DataFrame, nightly: float
) -> tuple[float | None, float | None]:
    """The band chosen on the development folds, applied to this one row."""
    method = artifact.get("interval_method") or "log_conformal"
    params = artifact.get("interval_params") or {}
    pred = np.array([nightly], dtype=float)
    if method == "scaled_log" and params:
        sigma = sigma_of(pred, float(params["a"]), float(params["b"]), float(params["floor"]))
        low, high = apply_log_radii(pred, float(params["quantile"]) * sigma)
    elif method == "asymmetric" and params:
        low, high = apply_signed_interval(pred, float(params["q_lo"]), float(params["q_hi"]))
    elif method == "mondrian_asymmetric" and params:
        low, high = apply_mondrian(pred, params)
    elif method == "cqr" and artifact.get("quantile_models") and params:
        models = artifact["quantile_models"]
        q_lo = predict_log(models["lo"], matrix, models["library"])
        q_hi = predict_log(models["hi"], matrix, models["library"])
        low, high = cqr_bounds(q_lo, q_hi, float(params["qhat"]))
    else:
        radius = artifact.get("conformal_log_radius")
        if radius is None:
            return None, None
        low, high = apply_log_interval(pred, float(radius))
    return float(low[0]), float(high[0])


def score_features(features: dict, city_key: str) -> dict:
    artifact = load_artifact(city_key)
    row = _frame_row(features, city_key, set(artifact["columns"]))
    supplied_text = bool(
        features.get("listing_text") or features.get("description") or features.get("name")
    )
    text_encoder = artifact.get("text_encoder")
    if text_encoder is not None and supplied_text:
        reduced = text_encoder.transform([str(row["listing_text"])])[0]
        for index, name in enumerate(text_encoder.column_names):
            row[name] = float(reduced[index])
    elif text_encoder is not None:
        for name in text_encoder.column_names:
            row[name] = np.nan
    embed = artifact.get("embed")
    if embed is not None and supplied_text:
        reduced = embed.transform([str(row["listing_text"])])[0]
        for index, name in enumerate(embed.column_names):
            value = float(reduced[index])
            row[name] = value if np.isfinite(value) else np.nan
    elif embed is not None:
        for name in embed.column_names:
            row[name] = np.nan
    comp = artifact.get("comp")
    if comp is not None:
        matched = comp.transform(
            np.array([float(features["latitude"])]),
            np.array([float(features["longitude"])]),
            np.array([row["room_type"]]),
            np.array([float(row["bedrooms"])]),
            row_index=None,
        )
        row["comp_log_price"] = float(matched[0])
    if "months_before" in set(artifact["columns"]):
        row["months_before"] = 0.0
    kernel = artifact.get("kernel")
    if kernel is not None:
        smoothed = kernel.transform(
            np.array([float(features["latitude"])]),
            np.array([float(features["longitude"])]),
            row_index=None,
        )
        row["kernel_log_price"] = float(smoothed[0])
    if artifact.get("pooled"):
        for name, value in city_dummies(city_key).items():
            row[name] = value
    columns = list(artifact["columns"])
    matrix = pd.DataFrame([{column: row.get(column, np.nan) for column in columns}])
    mode = artifact.get("spatial_mode") or ("both" if artifact.get("use_spatial") else "none")
    if mode in ("smooth", "knn", "both") and artifact.get("encoder") is not None:
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
    nightly_raw = _predict_matrix(artifact, matrix)
    nightly_adjusted = apply_correction(
        nightly_raw,
        artifact.get("correction") or "none",
        artifact.get("correction_params") or {},
    )
    nightly = float(nightly_adjusted[0])
    low, high = _interval(artifact, matrix, nightly)
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
        "interval_method": artifact.get("interval_method") or "log_conformal",
    }
