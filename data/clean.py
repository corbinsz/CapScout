"""Clean one Inside Airbnb listings file and record every cut.

The rules are fixed amounts and a fixed distance, not percentiles of the
file. A percentile computed before the split would let the held-out
listings help decide who counts as an outlier.

Price strings are the Inside Airbnb form `$1,234.00`. The parser removes
the dollar sign and the thousands comma, keeps a single decimal point, and
rejects anything with two decimal points. `Half-bath` in the bathroom text
is 0.5. A digit in that text fills a blank numeric bathroom.

Inactive listings are removed: no review in the last 12 months, or
unavailable every night of the next year. `availability_365` of 0 is the
second rule. A missing availability value is not treated as blocked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from evaluation.study_protocol import (
    MAX_BATHROOMS,
    MAX_BEDROOMS,
    MAX_GUESTS,
    MAX_KM_FROM_CENTER,
    MAX_MINIMUM_NIGHTS,
    MAX_PRICE,
    MIN_PRICE,
)
from features.geo import distance_to_city_center_km

from data.cities import CITIES
from data.ingest import listings_path
from data.occupancy import booked_nights_ltm

RAW_COLUMNS = [
    "id",
    "host_id",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_rating",
    "host_is_superhost",
    "amenities",
    "availability_365",
    "description",
    "name",
    "neighborhood_overview",
]


def source_record(city_key: str) -> dict:
    city = CITIES[city_key]
    url = (
        "https://data.insideairbnb.com/united-states/"
        f"{city['ia_state']}/{city['ia_city']}/{city['ia_date']}/data/listings.csv.gz"
    )
    return {
        "city": city_key,
        "display": city["display"],
        "snapshot": city["ia_date"],
        "url": url,
        "file": str(Path("data/raw/cities") / city_key / "listings.csv.gz"),
        "license": "CC BY 4.0",
        "credit": "Inside Airbnb",
        "catalog": "https://insideairbnb.com/get-the-data/",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
    }


def parse_price_series(series: pd.Series) -> pd.Series:
    """Parse `$1,234.00` to 1234.0. Two decimal points become missing."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    text = text.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    blank = text.str.lower().isin(["", "nan", "none", "null"])
    two_dots = text.str.count(r"\.") > 1
    parsed = pd.to_numeric(text.mask(blank), errors="coerce")
    return parsed.mask(two_dots)


def parse_bathrooms(numeric: pd.Series | None, text: pd.Series | None) -> pd.Series:
    """Prefer the numeric column. Otherwise read the text, and map half-baths to 0.5."""
    if numeric is None and text is None:
        raise ValueError("bathrooms need a numeric column or a text column")
    if text is None:
        return pd.to_numeric(numeric, errors="coerce")
    raw = text.fillna("").astype(str)
    half = raw.str.contains("half", case=False, na=False)
    extracted = pd.to_numeric(raw.str.extract(r"([0-9]*\.?[0-9]+)", expand=False), errors="coerce")
    extracted = extracted.mask(half & extracted.isna(), 0.5)
    if numeric is None:
        return extracted
    baths = pd.to_numeric(numeric, errors="coerce")
    return baths.fillna(extracted)


def _keep(df: pd.DataFrame, mask: np.ndarray, steps: list[dict], rule_id: str, description: str):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(df):
        raise RuntimeError(f"{rule_id} mask length {len(mask)} != {len(df)}")
    before = len(df)
    kept = df.iloc[np.flatnonzero(mask)].reset_index(drop=True)
    steps.append(
        {
            "id": rule_id,
            "description": description,
            "before": int(before),
            "removed": int(before - len(kept)),
            "remaining": int(len(kept)),
        }
    )
    return kept


def apply_filters(raw: pd.DataFrame, city_key: str) -> tuple[pd.DataFrame, dict]:
    """Return the modeling rows and a sequential log of every rule."""
    if city_key not in CITIES:
        raise KeyError(city_key)
    df = raw.copy().reset_index(drop=True)
    steps: list[dict] = [
        {
            "id": "file",
            "description": "Rows in the Inside Airbnb file",
            "before": int(len(df)),
            "removed": 0,
            "remaining": int(len(df)),
        }
    ]
    price_text = df["price"] if "price" in df.columns else pd.Series("", index=df.index)
    price_as_text = price_text.astype(str)
    notes = {
        "prices_with_dollar_sign": int(price_as_text.str.contains(r"\$", regex=True).sum()),
        "prices_with_thousands_comma": int(price_as_text.str.contains(",", regex=False).sum()),
    }

    if "id" not in df.columns:
        raise RuntimeError(f"{city_key} has no listing id")
    ltm = pd.to_numeric(df.get("number_of_reviews_ltm"), errors="coerce").fillna(0.0)
    listing_id = pd.to_numeric(df["id"], errors="coerce")
    # Higher last-year reviews first, then the lower id. Keep that one copy.
    order = np.lexsort((listing_id.fillna(np.inf).to_numpy(), -ltm.to_numpy()))
    seen_ids: set[str] = set()
    keep_id = np.zeros(len(df), dtype=bool)
    ids_as_str = df["id"].astype(str).to_numpy()
    for pos in order:
        key = ids_as_str[int(pos)]
        if key in seen_ids:
            continue
        seen_ids.add(key)
        keep_id[int(pos)] = True
    df = _keep(
        df,
        keep_id,
        steps,
        "duplicate_id",
        "Duplicate listing id (kept the copy with more reviews in the last year)",
    )

    parsed = parse_price_series(
        df["price"] if "price" in df.columns else pd.Series(np.nan, index=df.index)
    )
    df = df.copy()
    df["price_clean"] = parsed.to_numpy()
    finite_positive = parsed.notna().to_numpy() & (parsed.to_numpy() > 0)
    df = _keep(
        df,
        finite_positive,
        steps,
        "bad_price",
        "Price missing, not a number, or not positive",
    )
    df = _keep(
        df,
        df["price_clean"].to_numpy() >= MIN_PRICE,
        steps,
        "price_under_25",
        "Price under $25",
    )
    df = _keep(
        df,
        df["price_clean"].to_numpy() <= MAX_PRICE,
        steps,
        "price_over_2000",
        "Price over $2,000",
    )

    lat = pd.to_numeric(df.get("latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("longitude"), errors="coerce")
    df = _keep(
        df,
        lat.notna().to_numpy() & lon.notna().to_numpy(),
        steps,
        "missing_coordinates",
        "Missing latitude or longitude",
    )
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    distance = distance_to_city_center_km(
        df["latitude"].to_numpy(), df["longitude"].to_numpy(), city_key
    )
    df = _keep(
        df,
        np.asarray(distance) <= MAX_KM_FROM_CENTER,
        steps,
        "outside_city",
        "More than 60 km from downtown",
    )

    room = df["room_type"] if "room_type" in df.columns else pd.Series(np.nan, index=df.index)
    room_text = room.astype(str).str.strip()
    room_ok = room.notna().to_numpy() & ~room_text.str.lower().isin(["", "nan", "none"]).to_numpy()
    df = _keep(df, room_ok, steps, "missing_room_type", "Missing room type")

    bedrooms = pd.to_numeric(df.get("bedrooms"), errors="coerce")
    df = _keep(df, bedrooms.notna().to_numpy(), steps, "missing_bedrooms", "Missing bedrooms")
    bedrooms = pd.to_numeric(df["bedrooms"], errors="coerce")
    df = _keep(
        df,
        bedrooms.between(0, MAX_BEDROOMS).to_numpy(),
        steps,
        "bedroom_range",
        "Bedrooms outside 0 to 10",
    )

    numeric_baths = df["bathrooms"] if "bathrooms" in df.columns else None
    text_baths = df["bathrooms_text"] if "bathrooms_text" in df.columns else None
    raw_numeric = (
        pd.to_numeric(numeric_baths, errors="coerce")
        if numeric_baths is not None
        else pd.Series(np.nan, index=df.index)
    )
    baths = parse_bathrooms(numeric_baths, text_baths)
    notes["bathrooms_filled_from_text"] = int((raw_numeric.isna() & baths.notna()).sum())
    half_mask = (
        text_baths.fillna("").astype(str).str.contains("half", case=False, na=False)
        if text_baths is not None
        else pd.Series(False, index=df.index)
    )
    notes["half_baths_parsed"] = int((raw_numeric.isna() & half_mask & baths.notna()).sum())
    df = df.copy()
    df["bathrooms_clean"] = baths.to_numpy()
    bad_bath = baths.notna().to_numpy() & ~baths.between(0, MAX_BATHROOMS).to_numpy()
    df = _keep(
        df,
        ~bad_bath,
        steps,
        "bathroom_range",
        "Bathrooms outside 0 to 12",
    )

    guests = pd.to_numeric(df.get("accommodates"), errors="coerce")
    bad_guests = guests.notna().to_numpy() & ~guests.between(1, MAX_GUESTS).to_numpy()
    df = _keep(
        df,
        ~bad_guests,
        steps,
        "guest_range",
        "Guest capacity outside 1 to 16",
    )

    minimum = pd.to_numeric(df.get("minimum_nights"), errors="coerce")
    long_stay = minimum.notna().to_numpy() & (minimum.to_numpy() > MAX_MINIMUM_NIGHTS)
    df = _keep(
        df,
        ~long_stay,
        steps,
        "minimum_nights",
        "Minimum stay over 90 nights",
    )

    reviews = pd.to_numeric(df.get("number_of_reviews_ltm"), errors="coerce")
    df = _keep(
        df,
        reviews.fillna(0).to_numpy() >= 1,
        steps,
        "no_reviews_ltm",
        "No reviews in the last 12 months",
    )

    if "availability_365" in df.columns:
        available = pd.to_numeric(df["availability_365"], errors="coerce")
    else:
        available = pd.Series(np.nan, index=df.index)
    blocked = available.notna().to_numpy() & (available.to_numpy() == 0)
    df = _keep(
        df,
        ~blocked,
        steps,
        "blocked_all_year",
        "Unavailable every night of the year",
    )

    host = (
        df["host_id"].astype(str) if "host_id" in df.columns else pd.Series("nan", index=df.index)
    )
    lat_r = pd.to_numeric(df["latitude"], errors="coerce").round(4)
    lon_r = pd.to_numeric(df["longitude"], errors="coerce").round(4)
    room_key = df["room_type"].astype(str)
    bed_key = pd.to_numeric(df["bedrooms"], errors="coerce").round(1)
    price_key = df["price_clean"].round(2)
    ltm_now = pd.to_numeric(df.get("number_of_reviews_ltm"), errors="coerce").fillna(0)
    ids_now = pd.to_numeric(df["id"], errors="coerce").fillna(0)
    rank = np.lexsort((ids_now.to_numpy(), -ltm_now.to_numpy()))
    seen: set[str] = set()
    drop = np.zeros(len(df), dtype=bool)
    for pos in rank:
        key = (
            f"{host.iloc[pos]}|{lat_r.iloc[pos]:.4f}|{lon_r.iloc[pos]:.4f}|"
            f"{room_key.iloc[pos]}|{bed_key.iloc[pos]:.1f}|{price_key.iloc[pos]:.2f}"
        )
        if key in seen:
            drop[pos] = True
        else:
            seen.add(key)
    df = _keep(
        df,
        ~drop,
        steps,
        "near_duplicate",
        "Same host, spot, room type, bedrooms, and price (kept the reviewed copy)",
    )

    df["host_id"] = df["host_id"].astype(str) if "host_id" in df.columns else df["id"].astype(str)
    missing_host = df["host_id"].str.lower().isin(["", "nan", "none"])
    notes["host_id_filled_from_listing"] = int(missing_host.sum())
    df.loc[missing_host, "host_id"] = "listing-" + df.loc[missing_host, "id"].astype(str)

    guests = pd.to_numeric(df.get("accommodates"), errors="coerce")
    bedrooms = pd.to_numeric(df["bedrooms"], errors="coerce")
    missing_guests = guests.isna() | (guests <= 0)
    notes["accommodates_filled"] = int(missing_guests.sum())
    guests = guests.mask(missing_guests, np.maximum(bedrooms.to_numpy(), 1) * 2)
    df["accommodates"] = guests.to_numpy()
    df["bedrooms"] = bedrooms.to_numpy()
    df["bathrooms_clean"] = parse_bathrooms(
        df["bathrooms"] if "bathrooms" in df.columns else None,
        df["bathrooms_text"] if "bathrooms_text" in df.columns else None,
    ).to_numpy()
    df["minimum_nights"] = pd.to_numeric(df.get("minimum_nights"), errors="coerce")
    df["number_of_reviews"] = pd.to_numeric(df.get("number_of_reviews"), errors="coerce")
    df["number_of_reviews_ltm"] = pd.to_numeric(df.get("number_of_reviews_ltm"), errors="coerce")
    df["reviews_per_month"] = pd.to_numeric(df.get("reviews_per_month"), errors="coerce")
    df["review_scores_rating"] = pd.to_numeric(df.get("review_scores_rating"), errors="coerce")
    if "beds" in df.columns:
        df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    else:
        df["beds"] = np.nan
    if "neighbourhood_cleansed" not in df.columns:
        df["neighbourhood_cleansed"] = "unknown"
    df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].fillna("unknown").astype(str)
    if "property_type" not in df.columns:
        df["property_type"] = ""
    if "amenities" not in df.columns:
        df["amenities"] = np.nan
    if "host_is_superhost" not in df.columns:
        df["host_is_superhost"] = np.nan

    df["booked_nights"] = booked_nights_ltm(
        df["number_of_reviews_ltm"].to_numpy(),
        df["minimum_nights"].to_numpy(),
    )
    defined = np.isfinite(df["booked_nights"].to_numpy()) & np.isfinite(
        df["price_clean"].to_numpy()
    )
    df = _keep(
        df,
        defined,
        steps,
        "label_undefined",
        "Price or review-based nights could not be computed",
    )

    notes["final_missing_bathrooms"] = int(
        pd.to_numeric(df["bathrooms_clean"], errors="coerce").isna().sum()
    )
    notes["final_missing_beds"] = int(pd.to_numeric(df["beds"], errors="coerce").isna().sum())
    notes["final_missing_amenities"] = int(df["amenities"].isna().sum())
    notes["final_missing_review_score"] = int(df["review_scores_rating"].isna().sum())
    farthest = distance_to_city_center_km(
        df["latitude"].to_numpy(dtype=float),
        df["longitude"].to_numpy(dtype=float),
        city_key,
    )
    notes["farthest_listing_km"] = float(np.max(farthest)) if len(df) else 0.0
    notes["max_km_from_center"] = MAX_KM_FROM_CENTER
    notes["min_price"] = MIN_PRICE
    notes["max_price"] = MAX_PRICE

    df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    df = df.drop(columns=["availability_365"], errors="ignore")
    report = {
        **source_record(city_key),
        "steps": steps,
        "notes": notes,
        "rows_raw": int(steps[0]["remaining"]),
        "rows_final": int(len(df)),
    }
    return df, report


def load_clean_city(city_key: str) -> tuple[pd.DataFrame, dict]:
    path = listings_path(city_key)
    header = pd.read_csv(path, nrows=0).columns
    usecols = [column for column in RAW_COLUMNS if column in header]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    return apply_filters(raw, city_key)


def render_quality_markdown(reports: dict[str, dict]) -> str:
    """One funnel table. Remaining is the row count after that rule."""
    cities = list(reports)
    header = "| Filter | " + " | ".join(
        f"{reports[city]['display']} removed | {reports[city]['display']} left" for city in cities
    )
    rule = "| --- | " + " | ".join("---: | ---:" for _ in cities)
    lines = [
        "# Data quality",
        "",
        "Counts are sequential. Each rule runs on the rows the previous rule kept.",
        "The dollar cuts and the 60 km cut are fixed. They are not percentiles of the file.",
        "",
        header + " |",
        rule + " |",
    ]
    steps = reports[cities[0]]["steps"]
    for index, step in enumerate(steps):
        cells = [step["description"]]
        for city in cities:
            item = reports[city]["steps"][index]
            if item["id"] != step["id"]:
                raise RuntimeError(f"{city} filter {item['id']} != {step['id']}")
            cells.append(f"{item['removed']:,}")
            cells.append(f"{item['remaining']:,}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Sources", ""])
    for city in cities:
        record = reports[city]
        lines.append(
            f"- **{record['display']}** snapshot {record['snapshot']}, "
            f"{record['license']}, credit {record['credit']}. `{record['url']}`"
        )
    lines.extend(
        [
            "",
            "Inside Airbnb's catalog is https://insideairbnb.com/get-the-data/ "
            "and the license is https://creativecommons.org/licenses/by/4.0/.",
            "",
            "## Notes",
            "",
        ]
    )
    for city in cities:
        notes = reports[city]["notes"]
        lines.append(
            f"- **{reports[city]['display']}.** "
            f"{notes['prices_with_dollar_sign']:,} prices contained a dollar sign and "
            f"{notes['prices_with_thousands_comma']:,} contained a thousands comma. "
            f"At the bathroom step, {notes['bathrooms_filled_from_text']:,} bathrooms were "
            f"read from the text ({notes['half_baths_parsed']:,} of those were half-baths "
            f"set to 0.5). "
            f"The farthest kept listing is {notes['farthest_listing_km']:.1f} km from downtown. "
            f"Still missing at the end: {notes['final_missing_bathrooms']:,} bathrooms, "
            f"{notes['final_missing_beds']:,} beds, "
            f"{notes['final_missing_amenities']:,} amenity lists, "
            f"{notes['final_missing_review_score']:,} review scores."
        )
    if any("census" in reports[city] for city in cities):
        lines.extend(["", "## Census join", ""])
        lines.append(
            "ACS 2024 5-year tables B19013, B25064, B25077, and B01003, "
            "joined to TIGER/Line 2024 tracts. Public domain. "
            "A missing tract statistic is the median of the city's tracts, not a price."
        )
        lines.append("")
        for city in cities:
            census = reports[city].get("census")
            if not census:
                continue
            lines.append(
                f"- **{reports[city]['display']}.** "
                f"{census['polygon_fraction']:.1%} of listings fell in a tract polygon, "
                f"{census['centroid_fallback_fraction']:.1%} used a tract within 2 km, "
                f"{census['city_median_fraction']:.1%} used the city median. "
                f"{census['field_imputed_fraction']:.1%} had a missing tract statistic "
                "filled with the city median."
            )
    lines.append("")
    return "\n".join(lines)
