"""Columns that are functions of the listing, not of the target.

Neighbourhood price encodings and nearest-neighbour prices are NOT built
here. Those use other listings' prices, so they have to be fit inside each
training fold (see features.fold). Building them on the full city and then
splitting would leak the test listings' prices into the features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from data.cities import CITIES

from features.geo import distance_to_airport_km, distance_to_center_km
from features.text import compose_listing_text, lexicon_frame

# Exclusive property buckets. "other" is the dropped level (all flags zero),
# which keeps a linear model from having a dummy trap. Trees don't care.
PROPERTY_FLAGS = ("is_house", "is_cabin", "is_cottage", "is_condo", "is_townhouse", "is_guest")

# Entire home/apt is the baseline (all three zero). Room type is the product
# being rented; property flags above are the building.
ROOM_FLAGS = ("is_private_room", "is_shared_room", "is_hotel_room")

# Precomputed interactions. They do not use the target.
# A private room's capacity does not price like an entire home's capacity.
# Distance to downtown is allowed a different slope for houses than condos.
INTERACTION_COLUMNS = (
    "accommodates_x_private",
    "accommodates_x_shared",
    "distance_x_house",
    "distance_x_condo",
)

AMENITY_FLAGS = (
    "has_ac",
    "has_parking",
    "has_washer",
    "has_dryer",
    "has_pool",
    "has_hottub",
    "has_gym",
    "has_waterfront",
    "has_kitchen",
    "has_workspace",
)

# Added in the v3 pass. Kept separate so the v2 amenity block is unchanged.
EXTRA_AMENITY_FLAGS = (
    "has_fireplace",
    "has_dishwasher",
    "has_self_checkin",
    "has_patio",
    "has_bbq",
    "has_ev_charger",
    "allows_pets",
    "has_elevator",
)

_EXTRA_AMENITY_PATTERNS = {
    "has_fireplace": r"fireplace",
    "has_dishwasher": r"dishwasher",
    "has_self_checkin": r"self check-in|self checkin|smart lock|lockbox|keypad",
    "has_patio": r"patio|balcony",
    "has_bbq": r"\bbbq\b|barbecue|\bgrill\b",
    "has_ev_charger": r"ev charger|electric vehicle",
    "allows_pets": r"allows pets|pets allowed",
    "has_elevator": r"elevator",
}

# Inside Airbnb's four response-time labels, slowest last. Anything else
# (including a blank) is missing, not a zero.
_RESPONSE_TIME = {
    "within an hour": 1.0,
    "within a few hours": 2.0,
    "within a day": 3.0,
    "a few days or more": 4.0,
}

# Size, location, and property type. Known for a home that is not yet listed,
# once you know where it is. This is the feature set behind local scoring.
COMPS_BASE = [
    "bedrooms",
    "bathrooms",
    "accommodates",
    "latitude",
    "longitude",
    "distance_to_center_km",
    "distance_to_airport_km",
    *PROPERTY_FLAGS,
]

# Extra columns that describe an existing listing: how it is equipped, how
# many people have reviewed it, and who the host is. They help predict the
# advertised rate of a listing that already exists. They are not available
# for a house that has never been listed, so the scoring artifact does not
# require them.
LISTING_QUALITY = [
    "minimum_nights_capped",
    "amenity_count",
    *AMENITY_FLAGS,
    "review_score",
    "log_reviews",
    "reviews_per_month",
    "is_superhost",
    "log_host_listings",
    "instant_bookable",
]

FULL_BASE = COMPS_BASE + LISTING_QUALITY

# What the previous pipeline actually varied on. Sqft columns in that
# pipeline were entirely missing in the committed snapshots, so a constant
# median fill could not have helped. Cabin and cottage stay split out so
# the "before" model is not secretly a coarser feature set.
PREVIOUS_BASE = [
    "bedrooms",
    "bathrooms",
    "accommodates",
    "latitude",
    "longitude",
    "distance_to_center_km",
    *PROPERTY_FLAGS,
]


def _flag(text: pd.Series, pattern: str) -> pd.Series:
    return text.str.contains(pattern, regex=True, na=False).astype(np.int8)


def _property_flags(property_type: pd.Series) -> pd.DataFrame:
    text = property_type.fillna("").astype(str).str.lower()
    # Order matters: "guesthouse" contains "house", "townhouse" contains "house".
    guest = text.str.contains("guest", na=False)
    townhouse = text.str.contains("townhouse", na=False) & ~guest
    cabin = text.str.contains("cabin", na=False) & ~guest & ~townhouse
    cottage = text.str.contains("cottage", na=False) & ~guest & ~townhouse & ~cabin
    condo = text.str.contains("condo|apartment|rental unit|loft|serviced", regex=True, na=False)
    condo = condo & ~guest & ~townhouse & ~cabin & ~cottage
    house = text.str.contains("home|house|bungalow|villa|tiny|chalet|place", regex=True, na=False)
    house = house & ~guest & ~townhouse & ~cabin & ~cottage & ~condo
    return pd.DataFrame(
        {
            "is_house": house.astype(np.int8),
            "is_cabin": cabin.astype(np.int8),
            "is_cottage": cottage.astype(np.int8),
            "is_condo": condo.astype(np.int8),
            "is_townhouse": townhouse.astype(np.int8),
            "is_guest": guest.astype(np.int8),
        },
        index=property_type.index,
    )


def _amenity_frame(amenities: pd.Series) -> pd.DataFrame:
    raw = amenities.fillna("").astype(str)
    count = (raw.str.count('"') // 2).astype(np.int16)
    text = raw.str.lower().str.replace("pool table", " ", regex=False)
    return pd.DataFrame(
        {
            "amenity_count": count,
            "has_ac": _flag(text, r"air conditioning"),
            "has_parking": _flag(text, r"parking"),
            "has_washer": _flag(text, r"washer"),
            "has_dryer": _flag(text, r"dryer"),
            "has_pool": _flag(text, r"pool"),
            "has_hottub": _flag(text, r"hot tub|jacuzzi"),
            "has_gym": _flag(text, r"gym|exercise equipment"),
            "has_waterfront": _flag(text, r"waterfront|beachfront|beach access"),
            "has_kitchen": _flag(text, r"kitchen"),
            "has_workspace": _flag(text, r"workspace|laptop.friendly"),
        },
        index=amenities.index,
    )


def _percent_rate(series: pd.Series) -> pd.Series:
    """'97%' -> 0.97. A blank or 'N/A' stays missing."""
    text = series.astype(str).str.strip().str.replace("%", "", regex=False)
    values = pd.to_numeric(text, errors="coerce")
    values = values.where(values <= 1.0, values / 100.0)
    return values.where(values.between(0.0, 1.0))


def _yes_no(series: pd.Series) -> pd.Series:
    """t/f flags. An unrecognised cell is missing, not a no."""
    text = series.astype(str).str.lower().str.strip()
    known = text.isin(["t", "true", "1", "f", "false", "0"])
    values = text.isin(["t", "true", "1"]).astype(float)
    return values.where(known, np.nan)


def _review_aspect(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    # The file scores these on a 1–5 scale. A 0 or a 6 is a bad cell.
    return values.where(values.between(1.0, 5.0))


def _extra_amenities(amenities: pd.Series, missing: pd.Series) -> pd.DataFrame:
    text = amenities.fillna("").astype(str).str.lower()
    data = {
        name: text.str.contains(pattern, regex=True, na=False).astype(float)
        for name, pattern in _EXTRA_AMENITY_PATTERNS.items()
    }
    frame = pd.DataFrame(data, index=amenities.index)
    frame.loc[missing.to_numpy(), :] = np.nan
    return frame


def _attach_detail(out: pd.DataFrame, city_key: str) -> None:
    """Host, bed, bath, and review-aspect columns. Missing stays missing.

    These are not filled with zero. A home that has never been reviewed
    does not have a cleanliness score of 0, and a blank response rate is
    not a host who never answers.
    """
    beds = pd.to_numeric(out["beds"], errors="coerce") if "beds" in out.columns else np.nan
    if isinstance(beds, pd.Series):
        beds = beds.where(beds.between(0, 16))
        out["beds"] = beds.to_numpy()
    else:
        out["beds"] = np.nan

    if "bathrooms_text" in out.columns:
        bath_text = out["bathrooms_text"].astype(str)
        known = out["bathrooms_text"].notna() & ~bath_text.str.lower().isin(["nan", "none", ""])
        shared = bath_text.str.contains("shared", case=False, na=False).astype(float)
        out["bath_is_shared"] = shared.where(known, np.nan).to_numpy()
    else:
        out["bath_is_shared"] = np.nan

    as_of = pd.Timestamp(CITIES[city_key]["ia_date"])
    if "host_since" in out.columns:
        since = pd.to_datetime(out["host_since"], errors="coerce")
        years = (as_of - since).dt.days / 365.25
        years = years.where(years >= 0)
        out["log_host_tenure"] = np.log1p(years.to_numpy(dtype=float))
    else:
        out["log_host_tenure"] = np.nan

    if "host_response_rate" in out.columns:
        out["host_response_rate"] = _percent_rate(out["host_response_rate"]).to_numpy()
    else:
        out["host_response_rate"] = np.nan
    if "host_acceptance_rate" in out.columns:
        out["host_acceptance_rate"] = _percent_rate(out["host_acceptance_rate"]).to_numpy()
    else:
        out["host_acceptance_rate"] = np.nan
    if "host_response_time" in out.columns:
        labels = out["host_response_time"].astype(str).str.lower().str.strip()
        out["host_response_time_ord"] = labels.map(_RESPONSE_TIME).to_numpy(dtype=float)
    else:
        out["host_response_time_ord"] = np.nan
    if "host_identity_verified" in out.columns:
        out["host_identity_verified"] = _yes_no(out["host_identity_verified"]).to_numpy()
    else:
        out["host_identity_verified"] = np.nan

    aspects = {
        "review_cleanliness": "review_scores_cleanliness",
        "review_location": "review_scores_location",
        "review_accuracy": "review_scores_accuracy",
        "review_checkin": "review_scores_checkin",
        "review_communication": "review_scores_communication",
    }
    for dest, source in aspects.items():
        if source in out.columns:
            out[dest] = _review_aspect(out[source]).to_numpy()
        else:
            out[dest] = np.nan

    if "amenities" in out.columns:
        missing = out["amenities"].isna()
        extra = _extra_amenities(out["amenities"], missing)
    else:
        extra = _extra_amenities(pd.Series("", index=out.index), pd.Series(True, index=out.index))
    for column in extra.columns:
        out[column] = extra[column].to_numpy()


def attach_base_features(df: pd.DataFrame, city_key: str) -> pd.DataFrame:
    """Add leak-free columns. Does not drop the original listing fields."""
    out = df.copy()
    lat = out["latitude"].to_numpy(dtype=float)
    lon = out["longitude"].to_numpy(dtype=float)
    out["distance_to_center_km"] = distance_to_center_km(lat, lon, city_key)
    out["distance_to_airport_km"] = distance_to_airport_km(lat, lon, city_key)
    out["bathrooms"] = out["bathrooms_clean"] if "bathrooms_clean" in out.columns else np.nan

    flags = _property_flags(
        out["property_type"] if "property_type" in out.columns else pd.Series("", index=out.index)
    )
    for col in flags.columns:
        out[col] = flags[col].to_numpy()

    if "amenities" in out.columns:
        missing_amenity = out["amenities"].isna()
        amen = _amenity_frame(out["amenities"])
    else:
        missing_amenity = pd.Series(True, index=out.index)
        amen = _amenity_frame(pd.Series("", index=out.index))
    for col in amen.columns:
        # A blank amenity field is unknown, not "this place has no kitchen".
        # Known empty lists ("[]") still count as zero amenities.
        values = amen[col].astype(float)
        values = values.mask(missing_amenity, np.nan)
        out[col] = values.to_numpy()

    nights = pd.to_numeric(out.get("minimum_nights"), errors="coerce")
    # Cap at 90 for the same reason as the occupancy helper: a 365-night
    # "minimum" is not a meaningful nightly-rate feature, and leaving it
    # raw lets one bad cell dominate a split.
    out["minimum_nights_capped"] = nights.clip(lower=1, upper=90)

    reviews = pd.to_numeric(out.get("number_of_reviews"), errors="coerce").fillna(0)
    out["log_reviews"] = np.log1p(reviews.to_numpy(dtype=float))
    rpm = pd.to_numeric(out.get("reviews_per_month"), errors="coerce")
    # Missing reviews-per-month on this file means "no reviews yet", which
    # is a real zero, not a value we should median-fill into 1.5.
    out["reviews_per_month"] = rpm.fillna(0.0)
    out["review_score"] = pd.to_numeric(out.get("review_scores_rating"), errors="coerce")

    if "host_is_superhost" in out.columns:
        out["is_superhost"] = (
            out["host_is_superhost"]
            .astype(str)
            .str.lower()
            .isin(["t", "true", "1"])
            .astype(np.int8)
        )
    else:
        out["is_superhost"] = np.int8(0)
    if "instant_bookable" in out.columns:
        out["instant_bookable"] = (
            out["instant_bookable"].astype(str).str.lower().isin(["t", "true", "1"]).astype(np.int8)
        )
    else:
        out["instant_bookable"] = np.int8(0)
    if "host_listings_count" in out.columns:
        host_n = pd.to_numeric(out["host_listings_count"], errors="coerce")
    else:
        host_n = pd.Series(1.0, index=out.index)
    out["log_host_listings"] = np.log1p(
        host_n.clip(lower=0, upper=100).fillna(1.0).to_numpy(dtype=float)
    )

    if "room_type" in out.columns:
        room = out["room_type"].fillna("").astype(str)
    else:
        room = pd.Series("", index=out.index)
    out["is_private_room"] = room.eq("Private room").astype(np.int8)
    out["is_shared_room"] = room.eq("Shared room").astype(np.int8)
    out["is_hotel_room"] = room.str.contains("Hotel", case=False, na=False).astype(np.int8)
    out["accommodates_x_private"] = out["accommodates"] * out["is_private_room"]
    out["accommodates_x_shared"] = out["accommodates"] * out["is_shared_room"]
    out["distance_x_house"] = out["distance_to_center_km"] * out["is_house"]
    out["distance_x_condo"] = out["distance_to_center_km"] * out["is_condo"]
    _attach_detail(out, city_key)
    # The ad text. Lexicon columns are a function of this string only.
    # The TF-IDF compression is fit later, inside the training fold.
    out["listing_text"] = compose_listing_text(out)
    lexicon = lexicon_frame(out["listing_text"])
    for column in lexicon.columns:
        out[column] = lexicon[column].to_numpy()
    return out


def property_flags_from_label(label: str) -> dict[str, int]:
    """Map a for-sale property-type string onto the same exclusive flags."""
    text = (label or "").lower()
    flags = {name: 0 for name in PROPERTY_FLAGS}
    if "cabin" in text:
        flags["is_cabin"] = 1
    elif "cottage" in text:
        flags["is_cottage"] = 1
    elif "townhouse" in text:
        flags["is_townhouse"] = 1
    elif "condo" in text or "co-op" in text or "coop" in text:
        flags["is_condo"] = 1
    elif "guest" in text:
        flags["is_guest"] = 1
    elif any(
        token in text
        for token in ("single family", "multi-family", "mobile", "manufactured", "house", "home")
    ):
        flags["is_house"] = 1
    return flags
