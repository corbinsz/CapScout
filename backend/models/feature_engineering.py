import pandas as pd
import numpy as np
import re
import os
from sklearn.cluster import KMeans
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

    # --- Assessor enrichment (sqft, year_built, lot_size) ---
    enrichment_path = os.path.join(config.city_data_dir(city_key), 'assessor_enrichment.csv')
    if os.path.exists(enrichment_path):
        enrich = pd.read_csv(enrichment_path)
        df = df.merge(
            enrich[['listing_id', 'sqft', 'year_built', 'lot_sqft']],
            left_on='id', right_on='listing_id', how='left',
        )
        df.drop(columns=['listing_id'], inplace=True, errors='ignore')
        # Sanity-check ranges
        df.loc[df['sqft'] < 100, 'sqft'] = np.nan
        df.loc[df['sqft'] > 15000, 'sqft'] = np.nan
        df.loc[df['year_built'] < 1800, 'year_built'] = np.nan
        df.loc[df['year_built'] > 2026, 'year_built'] = np.nan
        df.loc[df['lot_sqft'] < 100, 'lot_sqft'] = np.nan
        df.loc[df['lot_sqft'] > 500000, 'lot_sqft'] = np.nan
        # Derived features
        df['sqft_per_bedroom'] = df['sqft'] / df['bedrooms'].clip(lower=1)
        df['property_age'] = 2025 - df['year_built']
        df['lot_acres'] = df['lot_sqft'] / 43560.0
        n_enriched = df['sqft'].notna().sum()
        print(f'  Assessor enrichment: {n_enriched}/{len(df)} listings with sqft')
    else:
        # No enrichment: create NaN columns so feature pipeline works
        for col in ['sqft', 'year_built', 'lot_sqft', 'sqft_per_bedroom',
                     'property_age', 'lot_acres']:
            df[col] = np.nan

    return df


def get_feature_columns():
    """Return the list of feature column names used by the model.

    Only includes features that are available (or derivable) for BOTH
    Airbnb training data AND for-sale prediction targets.  Airbnb-only
    features (review scores, host metrics, booking activity) are excluded
    because they get median-filled as constants for for-sale properties,
    adding noise without signal.
    """
    base = [
        'bedrooms', 'bathrooms_clean', 'accommodates', 'beds',
        'latitude', 'longitude', 'distance_to_center', 'amenity_count',
    ]
    amenity_cols = [f'has_{a.replace(" ", "_")}' for a in KEY_AMENITIES]
    property_type_cols = [f'is_{pt.replace("entire ", "").replace(" ", "_")}' for pt in PROPERTY_TYPES]
    # Physical property features from county assessor enrichment
    physical_cols = [
        'sqft', 'year_built', 'lot_sqft',
        'sqft_per_bedroom', 'property_age', 'lot_acres',
    ]
    return base + amenity_cols + property_type_cols + physical_cols


def prepare_features(df):
    """Extract feature matrix and targets from cleaned dataframe.

    Returns raw (unsmoothed) targets. Call smooth_targets() separately
    after splitting into train/test to avoid target leakage.
    """
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols].copy()
    # Two-step fill: median for columns with partial data, then 0 for all-NaN columns
    # (e.g. physical features in cities without assessor enrichment)
    X = X.fillna(X.median())
    X = X.fillna(0)

    targets = {}
    if 'price_clean' in df.columns:
        targets['price'] = df['price_clean'].values
    if 'occupancy_rate' in df.columns:
        targets['occupancy'] = df['occupancy_rate'].values

    # Listing IDs for group splitting and smoothing
    listing_ids = df['id'].values if 'id' in df.columns else np.arange(len(df))

    return X, targets, available_cols, listing_ids


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


def smooth_targets(X, y_price, y_occ, listing_ids, months=None,
                   train_mask=None, n_clusters=20, min_group=3):
    """Smooth targets to geographic area medians.

    Groups by (geographic_cluster x bedrooms [x month]).

    IMPORTANT: If train_mask is provided, clusters and medians are computed
    from TRAINING data only, then applied to all rows.  This prevents target
    leakage — test targets are defined by what the training data says the
    area median is, not by the test listing's own price.

    Args:
        X: Feature DataFrame (must have latitude, longitude, bedrooms cols)
        y_price: raw price target array
        y_occ: raw occupancy target array
        listing_ids: array of listing IDs
        months: optional int array (1-12) for monthly grouping
        train_mask: boolean array — True for training rows.  If None, uses
                    all data (only appropriate for final production model).
        n_clusters: number of geographic clusters
        min_group: minimum listings per group for a stable median

    Returns:
        (smoothed_y_price, smoothed_y_occ, n_smoothed, n_groups)
    """
    n = len(X)
    if n < 10 or 'latitude' not in X.columns or 'longitude' not in X.columns:
        return y_price.copy(), y_occ.copy(), 0, 0

    # Which rows to use for fitting clusters + computing medians
    if train_mask is not None:
        fit_idx = np.where(train_mask)[0]
    else:
        fit_idx = np.arange(n)

    # Build temp DataFrame for grouping
    tmp = pd.DataFrame({
        'latitude': X['latitude'].values,
        'longitude': X['longitude'].values,
        'bedrooms': X['bedrooms'].values if 'bedrooms' in X.columns else 2,
        'listing_id': listing_ids,
        'price': y_price,
        'occ': y_occ,
    })
    if months is not None:
        tmp['month'] = months

    # Cluster on unique TRAINING listing coordinates only
    fit_data = tmp.iloc[fit_idx]
    fit_coords = fit_data.groupby('listing_id')[['latitude', 'longitude']].first()

    if len(fit_coords) < 10:
        return y_price.copy(), y_occ.copy(), 0, 0

    n_clust = max(2, min(n_clusters, len(fit_coords) // 5))
    kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
    kmeans.fit(fit_coords.values)

    # Assign ALL rows to nearest training cluster
    tmp['_cluster'] = kmeans.predict(tmp[['latitude', 'longitude']].values)
    tmp['_bed_bin'] = tmp['bedrooms'].clip(upper=5).fillna(2).astype(int)

    group_cols = ['_cluster', '_bed_bin']
    if months is not None:
        group_cols.append('month')

    # Compute medians from TRAINING rows only
    fit_tmp = tmp.iloc[fit_idx]
    group_medians = fit_tmp.groupby(group_cols).agg(
        _med_price=('price', 'median'),
        _med_occ=('occ', 'median'),
        _group_size=('price', 'count'),
    ).reset_index()

    # Apply training-derived medians to ALL rows
    tmp = tmp.merge(group_medians, on=group_cols, how='left')

    has_data = tmp['_group_size'].fillna(0) >= min_group
    smoothed_price = y_price.copy()
    smoothed_occ = y_occ.copy()
    smoothed_price[has_data.values] = tmp.loc[has_data, '_med_price'].values
    smoothed_occ[has_data.values] = tmp.loc[has_data, '_med_occ'].values

    n_smoothed = int(has_data.sum())
    n_groups = int((group_medians['_group_size'] >= min_group).sum())
    total_groups = len(group_medians)
    label = 'train-only' if train_mask is not None else 'all-data'
    print(f'  Target smoothing ({label}): {n_smoothed}/{n} rows to area medians '
          f'({n_groups}/{total_groups} groups with {min_group}+ listings)')

    return smoothed_price, smoothed_occ, n_smoothed, n_groups


def prepare_monthly_data(df, calendar_monthly):
    """
    Create an expanded dataset: one row per listing x month.

    Merges property features with monthly occupancy AND price from calendar.
    If calendar lacks prices, falls back to listing-level price_clean.
    Adds cyclical month encoding (sin_month, cos_month).

    Returns raw (unsmoothed) targets.  Call smooth_targets() separately
    after splitting into train/test to avoid target leakage.

    Returns: X, y_price, y_occupancy, feature_cols, listing_ids, months
        listing_ids: numpy array for group-aware train/test splitting
        months: raw int months (1-12) for use by smooth_targets()
    """
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]

    if 'id' not in df.columns:
        return None, None, None, None, None, None

    # Include price_clean for fallback when calendar lacks prices
    keep_cols = ['id'] + available_cols
    if 'price_clean' in df.columns:
        keep_cols = ['id', 'price_clean'] + available_cols
    base = df[keep_cols].copy()
    base = base.rename(columns={'id': 'listing_id'})

    merged = base.merge(calendar_monthly, on='listing_id', how='inner')

    if len(merged) == 0:
        return None, None, None, None, None, None

    # Fill missing calendar prices with listing-level price
    if merged['avg_price'].isna().any() and 'price_clean' in merged.columns:
        merged['avg_price'] = merged['avg_price'].fillna(merged['price_clean'])

    # Drop any rows still without price
    merged = merged.dropna(subset=['avg_price'])
    merged = merged[merged['avg_price'] > 0]

    if len(merged) == 0:
        return None, None, None, None, None, None

    # Cap dataset size to keep training fast — sample by listing to keep
    # all months of each listing together
    MAX_ROWS = 50_000
    if len(merged) > MAX_ROWS:
        unique_ids = merged['listing_id'].unique()
        np.random.RandomState(42).shuffle(unique_ids)
        keep_ids = set()
        row_count = 0
        for lid in unique_ids:
            n = (merged['listing_id'] == lid).sum()
            if row_count + n > MAX_ROWS:
                break
            keep_ids.add(lid)
            row_count += n
        merged = merged[merged['listing_id'].isin(keep_ids)]

    # Save raw months before converting to sin/cos
    raw_months = merged['month'].values.copy()

    # Cyclical month encoding
    merged['sin_month'] = np.sin(2 * np.pi * merged['month'] / 12)
    merged['cos_month'] = np.cos(2 * np.pi * merged['month'] / 12)

    monthly_feature_cols = available_cols + ['sin_month', 'cos_month']
    X = merged[monthly_feature_cols].copy()
    X = X.fillna(X.median())
    X = X.fillna(0)
    y_price = merged['avg_price'].values
    y_occ = merged['occupancy_rate'].values
    listing_ids = merged['listing_id'].values

    return X, y_price, y_occ, monthly_feature_cols, listing_ids, raw_months


# Keep old function name as alias for backward compat
def prepare_monthly_occupancy_data(df, calendar_monthly):
    """Backward-compatible wrapper."""
    X, y_price, y_occ, feature_cols, _ids, _months = prepare_monthly_data(df, calendar_monthly)
    return X, y_occ, feature_cols
