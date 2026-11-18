"""Load an Inside Airbnb listings snapshot into one row per listing.

The modeling population is every listing with an advertised nightly price
between $25 and $2,000, including private and shared rooms. Room type is a
feature, and the baseline median is also computed by room type, so the
model is not credited for a split the baseline already makes. The dollar
bounds are fixed in advance, not city percentiles: a percentile computed
on the whole file would be influenced by rows that later land in the test
fold.

Square footage is not on these files. Earlier assessor joins were never
committed with the snapshots, so this study does not invent sqft.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from features.frame import attach_base_features

from data.cities import CITIES, get_city_or_default
from data.occupancy import san_francisco_occupancy

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "cities"

MIN_PRICE = 25.0
MAX_PRICE = 2000.0

_WANTED = [
    "id",
    "name",
    "listing_url",
    "picture_url",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "host_is_superhost",
    "host_listings_count",
    "instant_bookable",
    "amenities",
    "host_id",
]


def raw_city_dir(city_key: str) -> Path:
    path = RAW_ROOT / city_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def listings_path(city_key: str) -> Path:
    directory = raw_city_dir(city_key)
    gz = directory / "listings.csv.gz"
    if gz.exists():
        return gz
    plain = directory / "listings.csv"
    if plain.exists():
        return plain
    raise FileNotFoundError(
        f"No listings file in {directory}. Download one with: python -m data.download {city_key}"
    )


def _parse_price(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(r"[^0-9.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _bathrooms(df: pd.DataFrame) -> pd.Series:
    if "bathrooms" in df.columns:
        baths = pd.to_numeric(df["bathrooms"], errors="coerce")
    else:
        baths = pd.Series(np.nan, index=df.index, dtype=float)
    if "bathrooms_text" in df.columns:
        extracted = df["bathrooms_text"].astype(str).str.extract(r"([0-9]*\.?[0-9]+)", expand=False)
        baths = baths.fillna(pd.to_numeric(extracted, errors="coerce"))
    return baths


def load_listings(city_key: str, *, entire_home_only: bool = False) -> tuple[pd.DataFrame, dict]:
    """Return the modeling frame and a count of how many rows each rule removed.

    `occupancy_rate` is the San Francisco review proxy (NaN when the listing
    has no reviews). It is not a training target for the price model.
    """
    if city_key not in CITIES:
        raise KeyError(f"Unknown city: {city_key}")

    path = listings_path(city_key)
    header = pd.read_csv(path, nrows=0).columns
    usecols = [col for col in _WANTED if col in header]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    stats: dict = {
        "city": city_key,
        "snapshot": CITIES[city_key]["ia_date"],
        "rows_raw": int(len(raw)),
        "entire_home_only": entire_home_only,
        "min_price": MIN_PRICE,
        "max_price": MAX_PRICE,
    }

    df = raw.copy()
    if entire_home_only and "room_type" in df.columns:
        before = len(df)
        df = df[df["room_type"] == "Entire home/apt"].copy()
        stats["dropped_not_entire_home"] = int(before - len(df))
    else:
        stats["dropped_not_entire_home"] = 0

    df["price_clean"] = _parse_price(df["price"]) if "price" in df.columns else np.nan
    price = df["price_clean"].to_numpy(dtype=float)
    finite = np.isfinite(price)
    stats["dropped_missing_or_nonpositive_price"] = int((~finite | (price <= 0)).sum())
    stats["dropped_below_min_price"] = int((finite & (price > 0) & (price < MIN_PRICE)).sum())
    stats["dropped_above_max_price"] = int((finite & (price > MAX_PRICE)).sum())
    df = df[df["price_clean"].between(MIN_PRICE, MAX_PRICE)].copy()

    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    stats["dropped_missing_coordinates"] = int(before - len(df))

    df["bedrooms"] = pd.to_numeric(df.get("bedrooms"), errors="coerce")
    stats["dropped_missing_bedrooms"] = int(df["bedrooms"].isna().sum())
    stats["dropped_bedroom_range"] = int(
        (df["bedrooms"].notna() & ~df["bedrooms"].between(0, 10)).sum()
    )
    df = df[df["bedrooms"].between(0, 10)].copy()

    if "room_type" in df.columns:
        missing_room = int(df["room_type"].isna().sum())
        df = df[df["room_type"].notna()].copy()
    else:
        missing_room = 0
        df["room_type"] = "Entire home/apt"
    stats["dropped_missing_room_type"] = missing_room

    if "host_id" in df.columns:
        df["host_id"] = df["host_id"].astype(str)
    else:
        df["host_id"] = (
            df["id"].astype(str) if "id" in df.columns else np.arange(len(df)).astype(str)
        )

    df["bathrooms_clean"] = _bathrooms(df)
    df["accommodates"] = pd.to_numeric(df.get("accommodates"), errors="coerce")
    # A missing guest count is a hole in the file, not a zero. Fill from
    # bedrooms only so train and a for-sale home (which has no true
    # accommodates) use the same rule: two guests per bedroom.
    missing_acc = df["accommodates"].isna() | (df["accommodates"] <= 0)
    df.loc[missing_acc, "accommodates"] = df.loc[missing_acc, "bedrooms"] * 2

    df["minimum_nights"] = pd.to_numeric(df.get("minimum_nights"), errors="coerce")
    df["number_of_reviews"] = pd.to_numeric(df.get("number_of_reviews"), errors="coerce").fillna(0)
    df["reviews_per_month"] = pd.to_numeric(df.get("reviews_per_month"), errors="coerce")
    df["review_scores_rating"] = pd.to_numeric(df.get("review_scores_rating"), errors="coerce")
    if "neighbourhood_cleansed" not in df.columns:
        df["neighbourhood_cleansed"] = "unknown"
    df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].fillna("unknown").astype(str)

    city = get_city_or_default(city_key)
    df["occupancy_rate"] = san_francisco_occupancy(
        df["reviews_per_month"].to_numpy(),
        df["minimum_nights"].fillna(city["avg_stay_nights"]).to_numpy(),
        float(city["avg_stay_nights"]),
    )

    df = attach_base_features(df, city_key)
    df = df.reset_index(drop=True)
    stats["rows_final"] = int(len(df))
    stats["occupancy_observed"] = int(np.isfinite(df["occupancy_rate"]).sum())
    stats["room_type_counts"] = {str(k): int(v) for k, v in df["room_type"].value_counts().items()}
    stats["n_hosts"] = int(df["host_id"].nunique())
    return df, stats


def load_study_frame(city_key: str) -> tuple[pd.DataFrame, dict]:
    """Modeling frame: cleaned listings plus census and OpenStreetMap columns.

    The external columns are a function of coordinates. They are attached
    before the split because they do not use the price; the split audit
    checks that no price column other than the fold-fit encodings is used.
    """
    from features.place import attach_place_features

    frame, stats = load_listings(city_key)
    frame, info = attach_place_features(frame, city_key)
    stats["place"] = info
    return frame, stats


def load_and_clean_listings(city_key: str | None = None) -> pd.DataFrame:
    """Frame used by the local app's analytics views."""
    frame, _stats = load_listings(city_key or "austin")
    return frame
