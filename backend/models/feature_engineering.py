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


def load_and_clean_listings():
    """Load Inside Airbnb listings.csv and clean it."""
    csv_path = os.path.join(config.DATA_RAW_DIR, 'listings.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"listings.csv not found at {csv_path}. "
            "Download from insideairbnb.com/get-the-data (Austin, TX)."
        )

    df = pd.read_csv(csv_path, low_memory=False)

    # Filter to entire homes/apartments only
    if 'room_type' in df.columns:
        df = df[df['room_type'] == 'Entire home/apt'].copy()

    # Clean price
    if 'price' in df.columns:
        df['price_clean'] = df['price'].apply(clean_price)
        df = df[df['price_clean'].notna() & (df['price_clean'] > 0)]
        # Remove outliers (below 1st percentile, above 99th)
        p1 = df['price_clean'].quantile(0.01)
        p99 = df['price_clean'].quantile(0.99)
        df = df[(df['price_clean'] >= p1) & (df['price_clean'] <= p99)]

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

    # Review scores
    for col in ['review_scores_rating', 'review_scores_cleanliness',
                'review_scores_location', 'review_scores_value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if df[col].notna().any() else 4.0)

    # Latitude / Longitude
    df['latitude'] = pd.to_numeric(df.get('latitude'), errors='coerce')
    df['longitude'] = pd.to_numeric(df.get('longitude'), errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])

    return df


def get_feature_columns():
    """Return the list of feature column names used by the model."""
    base = [
        'bedrooms', 'bathrooms_clean', 'accommodates', 'beds',
        'latitude', 'longitude', 'amenity_count',
        'review_scores_rating', 'review_scores_cleanliness',
        'review_scores_location', 'review_scores_value',
    ]
    amenity_cols = [f'has_{a.replace(" ", "_")}' for a in KEY_AMENITIES]
    return base + amenity_cols


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
