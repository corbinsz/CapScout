"""TreeSHAP contributions in log-price units.

The bias column is dropped. A positive value pushed the prediction up
relative to the model's baseline. For a single row the app turns that into
an approximate percent, exp(shap) - 1, which is the right scale because the
model was trained on log1p(price).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from features.columns import pretty


def shap_matrix(model, frame: pd.DataFrame, library: str) -> np.ndarray:
    frame = frame.reset_index(drop=True)
    if library == "xgboost":
        import xgboost as xgb

        contribs = model.get_booster().predict(xgb.DMatrix(frame), pred_contribs=True)
    elif library == "lightgbm":
        contribs = model.predict(frame, pred_contrib=True)
    elif library == "catboost":
        from catboost import Pool

        contribs = model.get_feature_importance(Pool(frame), type="ShapValues")
    else:
        raise ValueError(library)
    values = np.asarray(contribs, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values[:, :-1]


def top_reasons(
    names: list[str],
    values: np.ndarray,
    k: int = 5,
    *,
    observed: np.ndarray | None = None,
    feature_values: np.ndarray | None = None,
) -> list[dict]:
    """Largest contributions. Skip columns the caller left missing.

    A missing review count is not "this host has few listings." Showing that
    SHAP bar would describe the hole, not the home.
    """
    order = np.argsort(-np.abs(values))
    rows = []
    for index in order:
        if observed is not None and not bool(observed[index]):
            continue
        shap = float(values[index])
        if shap == 0.0:
            continue
        entry = {
            "feature": names[int(index)],
            "shap": shap,
            "approx_percent": float((np.exp(shap) - 1.0) * 100.0),
        }
        if feature_values is not None:
            entry["value"] = float(feature_values[index])
        rows.append(entry)
        if len(rows) >= k:
            break
    return rows


def collapse_wording(
    names: list[str],
    values: np.ndarray,
    observed: np.ndarray | None = None,
    feature_values: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Add the TF-IDF components into one signed 'listing wording' reason.

    The components are a compression of the ad. Separately they are not a
    sentence. The sum is the net push from the wording.
    """
    kept_names: list[str] = []
    kept_values: list[float] = []
    kept_observed: list[bool] = []
    kept_features: list[float] = []
    wording = 0.0
    wording_seen = False
    wording_observed = False
    for index, name in enumerate(names):
        if str(name).startswith("text_svd_"):
            wording += float(values[index])
            wording_seen = True
            if observed is None or bool(observed[index]):
                wording_observed = True
            continue
        kept_names.append(str(name))
        kept_values.append(float(values[index]))
        if observed is not None:
            kept_observed.append(bool(observed[index]))
        if feature_values is not None:
            kept_features.append(float(feature_values[index]))
    if wording_seen:
        kept_names.append("listing_wording")
        kept_values.append(wording)
        if observed is not None:
            kept_observed.append(wording_observed)
        if feature_values is not None:
            kept_features.append(float("nan"))
    observed_out = np.asarray(kept_observed, dtype=bool) if observed is not None else None
    features_out = np.asarray(kept_features, dtype=float) if feature_values is not None else None
    return kept_names, np.asarray(kept_values, dtype=float), observed_out, features_out


def _number(value: float, digits: int = 0) -> str:
    if digits == 0:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def _fact(feature: str, value: float | None) -> str:
    """The part of the sentence that names the fact, before the direction."""
    missing = value is None or (isinstance(value, float) and not math.isfinite(value))
    if feature == "bedrooms" and not missing:
        count = int(round(value))
        if count <= 0:
            return "A studio, with no separate bedroom,"
        noun = "bedroom" if count == 1 else "bedrooms"
        return f"{count} {noun}"
    if feature == "bathrooms" and not missing:
        noun = "bathroom" if value == 1 else "bathrooms"
        return f"{value:g} {noun}"
    if feature == "beds" and not missing:
        count = int(round(value))
        noun = "bed" if count == 1 else "beds"
        return f"{count} {noun}"
    if feature == "accommodates" and not missing:
        return f"Room for {int(round(value))} guests"
    if feature == "distance_to_center_km" and not missing:
        if value < 0.15:
            return "Sitting downtown"
        return f"Being {value:.1f} km from downtown"
    if feature == "distance_to_airport_km" and not missing:
        return f"Being {value:.1f} km from the airport"
    if (
        feature in {"census_median_income", "census_median_rent", "census_median_home_value"}
        and not missing
    ):
        return f"A tract {pretty(feature).lower()} of ${_number(value)}"
    if feature == "census_pop_density" and not missing:
        return f"A tract density of {_number(value)} people per km²"
    if feature == "reviews_per_month" and not missing:
        return f"{value:.1f} reviews a month"
    if feature == "review_score" and not missing:
        return f"A review score of {value:.2f}"
    if (
        feature
        in {
            "review_cleanliness",
            "review_location",
            "review_accuracy",
            "review_checkin",
            "review_communication",
        }
        and not missing
    ):
        return f"A {pretty(feature).lower()} of {value:.1f}"
    if feature == "minimum_nights_capped" and not missing:
        nights = int(round(value))
        noun = "night" if nights == 1 else "nights"
        return f"A {nights}-{noun} minimum"
    if feature == "host_response_rate" and not missing:
        return f"A host response rate of {value:.0%}"
    if feature == "host_acceptance_rate" and not missing:
        return f"A host acceptance rate of {value:.0%}"
    if feature == "log_reviews":
        return "The number of reviews"
    if feature == "log_host_listings":
        return "The size of the host's portfolio"
    if feature == "log_host_tenure":
        return "How long the host has been hosting"
    if feature == "log_text_chars":
        return "The length of the listing text"
    if feature == "listing_wording":
        return "The wording of the listing"
    if feature == "kernel_log_price":
        return "Prices of other listings a few kilometres away"
    if feature in {"latitude", "longitude"}:
        return "Where the home sits in the city"
    if feature == "is_private_room":
        if not missing and value >= 0.5:
            return "A private room, rather than an entire home,"
        return "An entire home, rather than a private room,"
    if feature == "is_shared_room":
        if not missing and value >= 0.5:
            return "A shared room"
        return "Not being a shared room"
    if feature == "bath_is_shared":
        if not missing and value >= 0.5:
            return "A shared bathroom"
        return "A private bathroom"
    if feature.startswith("text_mentions_") and not missing:
        label = pretty(feature).removeprefix("Text mentions ")
        if value >= 0.5:
            return f"The listing mentions {label}, which"
        return f"The listing does not mention {label}, which"
    if (
        feature.startswith("has_")
        or feature.startswith("allows_")
        or feature == "instant_bookable"
        or feature == "is_superhost"
        or feature == "host_identity_verified"
    ) and not missing:
        label = pretty(feature).lower()
        if value >= 0.5:
            return f"{pretty(feature)}"
        return f"No {label}"
    if feature.startswith("city_"):
        return f"Being in {pretty(feature)}"
    return pretty(feature)


def plain_reason(feature: str, value: float | None, percent: float) -> str:
    """One sentence for the app. The percent is shown beside it, not in it."""
    direction = "pushes the rate up" if percent >= 0 else "pushes the rate down"
    fact = _fact(feature, value)
    if fact.endswith(","):
        return f"{fact} {direction}."
    return f"{fact} {direction}."
