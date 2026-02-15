import pandas as pd
import numpy as np
import re
import os
import config


KEY_AMENITIES = [
    'pool', 'hot tub', 'wifi', 'kitchen', 'free parking',
    'air conditioning', 'washer', 'dryer', 'gym', 'ev charger',
    'self check-in', 'dedicated workspace', 'fireplace', 'bbq',
    'outdoor dining', 'patio', 'lake access', 'waterfront'
]

# Top property types to create binary flags for
PROPERTY_TYPES = [
    'entire home', 'entire rental unit', 'entire condo',
    'entire townhouse', 'entire guesthouse', 'entire guest suite',
    'entire cabin', 'entire cottage',
]


def clean_price(price_str):
    """Convert '$1,234.56' to float."""
    if pd.isna(price_str):
        return np.nan
    if isinstance(price_str, (int, float)):
        return float(price_str)
    return float(re.sub(r'[^\d.]', '', str(price_str)))


def parse_bathrooms(row):
    """Extract bathroom count from bathrooms_text or bathrooms column."""
    if 'bathrooms' in row and pd.notna(row.get('bathrooms')):
        return float(row['bathrooms'])
    text = row.get('bathrooms_text', '')
    if pd.isna(text) or not text:
        return 1.0
    match = re.search(r'([\d.]+)', str(text))
    return float(match.group(1)) if match else 1.0


def count_amenities(amenities_str):
    """Count total amenities from the amenities JSON-like string."""
    if pd.isna(amenities_str):
        return 0
    return len(re.findall(r'"([^"]+)"', str(amenities_str)))


def has_amenity(amenities_str, keyword):
    """Check if a specific amenity keyword exists."""
    if pd.isna(amenities_str):
        return 0
    return 1 if keyword.lower() in str(amenities_str).lower() else 0


def parse_pct(val):
    """Convert '95%' to 0.95, NaN to NaN."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace('%', '').strip()
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_distance_to_center(df, city_key=None):
    """Add distance_to_center column (km from city center)."""
    from cities import get_city_or_default
    city = get_city_or_default(city_key)
    df['distance_to_center'] = haversine_km(
        df['latitude'].values, df['longitude'].values,
        city['latitude'], city['longitude'],
    )
    return df


def load_and_clean_listings(city_key=None):
    """Load Inside Airbnb listings.csv and clean it."""
    from cities import DEFAULT_CITY
    city_key = city_key or DEFAULT_CITY
    csv_path = os.path.join(config.city_data_dir(city_key), 'listings.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"listings.csv not found at {csv_path}. "
            "Run: python -m scripts.download_data"
        )

    df = pd.read_csv(csv_path, low_memory=False)

    # Filter to entire homes/apartments only
    if 'room_type' in df.columns:
        df = df[df['room_type'] == 'Entire home/apt'].copy()

    # Clean price
    if 'price' in df.columns:
        df['price_clean'] = df['price'].apply(clean_price)
        df = df[df['price_clean'].notna() & (df['price_clean'] > 0)]
        # Remove outliers (below 2nd percentile, above 98th)
        p2 = df['price_clean'].quantile(0.02)
        p98 = df['price_clean'].quantile(0.98)
        df = df[(df['price_clean'] >= p2) & (df['price_clean'] <= p98)]

    # Parse bathrooms
    df['bathrooms_clean'] = df.apply(parse_bathrooms, axis=1)

    # Derive occupancy from availability_30
    if 'availability_30' in df.columns:
        df['occupancy_rate'] = 1 - (df['availability_30'] / 30.0)
        df['occupancy_rate'] = df['occupancy_rate'].clip(0, 1)

    # Amenity features
    if 'amenities' in df.columns:
        df['amenity_count'] = df['amenities'].apply(count_amenities)
        for amenity in KEY_AMENITIES:
            col_name = f'has_{amenity.replace(" ", "_")}'
            df[col_name] = df['amenities'].apply(lambda x, a=amenity: has_amenity(x, a))

    # Fill missing numeric columns
    for col in ['bedrooms', 'beds', 'accommodates']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if df[col].notna().any() else 1)

    # Review scores (expanded set)
    review_cols = [
        'review_scores_rating', 'review_scores_cleanliness',
        'review_scores_location', 'review_scores_value',
        'review_scores_accuracy', 'review_scores_checkin',
        'review_scores_communication',
    ]
    for col in review_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if df[col].notna().any() else 4.0)

    # --- New features ---

    # Property type binary flags
    if 'property_type' in df.columns:
        pt_lower = df['property_type'].str.lower().fillna('')
        for pt in PROPERTY_TYPES:
            col_name = f'is_{pt.replace("entire ", "").replace(" ", "_")}'
            df[col_name] = (pt_lower == pt).astype(int)

    # Host features
    if 'host_is_superhost' in df.columns:
        df['is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
    if 'instant_bookable' in df.columns:
        df['is_instant_bookable'] = (df['instant_bookable'] == 't').astype(int)
    if 'host_response_rate' in df.columns:
        df['host_response_pct'] = df['host_response_rate'].apply(parse_pct)
    if 'host_acceptance_rate' in df.columns:
        df['host_acceptance_pct'] = df['host_acceptance_rate'].apply(parse_pct)
    if 'host_since' in df.columns:
        df['days_as_host'] = (pd.Timestamp.now() - pd.to_datetime(df['host_since'], errors='coerce')).dt.days
        df['days_as_host'] = df['days_as_host'].clip(lower=0).fillna(0)
    if 'host_total_listings_count' in df.columns:
        df['host_listings'] = pd.to_numeric(df['host_total_listings_count'], errors='coerce').clip(upper=50).fillna(1)

    # Booking constraints
    if 'minimum_nights' in df.columns:
        df['min_nights'] = pd.to_numeric(df['minimum_nights'], errors='coerce').clip(upper=365).fillna(1)

    # Review activity (proxy for demand)
    if 'number_of_reviews' in df.columns:
        df['num_reviews'] = pd.to_numeric(df['number_of_reviews'], errors='coerce').fillna(0)
    if 'reviews_per_month' in df.columns:
        df['reviews_per_month_clean'] = pd.to_numeric(df['reviews_per_month'], errors='coerce').fillna(0)
    if 'number_of_reviews_ltm' in df.columns:
        df['reviews_ltm'] = pd.to_numeric(df['number_of_reviews_ltm'], errors='coerce').fillna(0)

    # Availability
    if 'availability_365' in df.columns:
        df['avail_365'] = pd.to_numeric(df['availability_365'], errors='coerce').fillna(180)

    # Latitude / Longitude
    df['latitude'] = pd.to_numeric(df.get('latitude'), errors='coerce')
    df['longitude'] = pd.to_numeric(df.get('longitude'), errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])

    # Distance to city center
    df = add_distance_to_center(df, city_key)

    return df


def get_feature_columns():
    """Return the list of feature column names used by the model."""
    base = [
        'bedrooms', 'bathrooms_clean', 'accommodates', 'beds',
        'latitude', 'longitude', 'distance_to_center', 'amenity_count',
        'review_scores_rating', 'review_scores_cleanliness',
        'review_scores_location', 'review_scores_value',
        'review_scores_accuracy', 'review_scores_checkin',
        'review_scores_communication',
    ]
    amenity_cols = [f'has_{a.replace(" ", "_")}' for a in KEY_AMENITIES]
    property_type_cols = [f'is_{pt.replace("entire ", "").replace(" ", "_")}' for pt in PROPERTY_TYPES]
    host_cols = [
        'is_superhost', 'is_instant_bookable',
        'host_response_pct', 'host_acceptance_pct',
        'days_as_host', 'host_listings',
    ]
    activity_cols = [
        'min_nights', 'num_reviews', 'reviews_per_month_clean',
        'reviews_ltm', 'avail_365',
    ]
    return base + amenity_cols + property_type_cols + host_cols + activity_cols


def prepare_features(df):
    """Extract feature matrix and targets from cleaned dataframe."""
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols].copy()
    X = X.fillna(X.median())

    targets = {}
    if 'price_clean' in df.columns:
        targets['price'] = df['price_clean'].values
    if 'occupancy_rate' in df.columns:
        targets['occupancy'] = df['occupancy_rate'].values

    return X, targets, available_cols


# ---------------------------------------------------------------------------
# Calendar-based monthly data (price + occupancy)
# ---------------------------------------------------------------------------

def load_calendar_monthly(city_key=None):
    """
    Load calendar.csv.gz and compute per-listing per-month stats.

    Returns DataFrame with columns:
        listing_id, month, occupancy_rate, avg_price

    If the calendar lacks price data (newer Inside Airbnb snapshots),
    returns occupancy-only data with avg_price=NaN.  The caller
    (prepare_monthly_data) will fill prices from the listing itself.
    """
    from cities import DEFAULT_CITY
    base = config.city_data_dir(city_key or DEFAULT_CITY)
    csv_path = os.path.join(base, 'calendar.csv.gz')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base, 'calendar.csv')
    if not os.path.exists(csv_path):
        return None

    cal = pd.read_csv(csv_path, usecols=['listing_id', 'date', 'available', 'price'],
                      low_memory=False)

    # Need at least some rows
    if len(cal) < 100:
        return None

    cal['date'] = pd.to_datetime(cal['date'])
    cal['month'] = cal['date'].dt.month
    cal['booked'] = (cal['available'] == 'f').astype(int)
    cal['price_clean'] = cal['price'].apply(clean_price)

    has_prices = cal['price_clean'].notna().sum() > len(cal) * 0.1

    if has_prices:
        monthly = cal.groupby(['listing_id', 'month']).agg(
            occupancy_rate=('booked', 'mean'),
            avg_price=('price_clean', 'mean'),
        ).reset_index()
        monthly = monthly.dropna(subset=['avg_price'])
        monthly = monthly[monthly['avg_price'] > 0]
    else:
        # Calendar has availability but no prices — still useful for occupancy
        monthly = cal.groupby(['listing_id', 'month']).agg(
            occupancy_rate=('booked', 'mean'),
        ).reset_index()
        monthly['avg_price'] = np.nan  # will be filled from listing price

    return monthly


def prepare_monthly_data(df, calendar_monthly):
    """
    Create an expanded dataset: one row per listing x month.

    Merges property features with monthly occupancy AND price from calendar.
    If calendar lacks prices, falls back to listing-level price_clean.
    Adds cyclical month encoding (sin_month, cos_month).

    Returns: X, y_price, y_occupancy, feature_cols
    """
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]

    if 'id' not in df.columns:
        return None, None, None, None

    # Include price_clean for fallback when calendar lacks prices
    keep_cols = ['id'] + available_cols
    if 'price_clean' in df.columns:
        keep_cols = ['id', 'price_clean'] + available_cols
    base = df[keep_cols].copy()
    base = base.rename(columns={'id': 'listing_id'})

    merged = base.merge(calendar_monthly, on='listing_id', how='inner')

    if len(merged) == 0:
        return None, None, None, None

    # Fill missing calendar prices with listing-level price
    if merged['avg_price'].isna().any() and 'price_clean' in merged.columns:
        merged['avg_price'] = merged['avg_price'].fillna(merged['price_clean'])

    # Drop any rows still without price
    merged = merged.dropna(subset=['avg_price'])
    merged = merged[merged['avg_price'] > 0]

    if len(merged) == 0:
        return None, None, None, None

    # Cap dataset size to keep training fast
    MAX_ROWS = 50_000
    if len(merged) > MAX_ROWS:
        merged = merged.sample(n=MAX_ROWS, random_state=42)

    # Cyclical month encoding
    merged['sin_month'] = np.sin(2 * np.pi * merged['month'] / 12)
    merged['cos_month'] = np.cos(2 * np.pi * merged['month'] / 12)

    monthly_feature_cols = available_cols + ['sin_month', 'cos_month']
    X = merged[monthly_feature_cols].copy()
    X = X.fillna(X.median())
    y_price = merged['avg_price'].values
    y_occ = merged['occupancy_rate'].values

    return X, y_price, y_occ, monthly_feature_cols


# Keep old function name as alias for backward compat
def prepare_monthly_occupancy_data(df, calendar_monthly):
    """Backward-compatible wrapper."""
    X, y_price, y_occ, feature_cols = prepare_monthly_data(df, calendar_monthly)
    return X, y_occ, feature_cols
