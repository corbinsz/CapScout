"""
Inside Airbnb data service — provides comparable listings, neighborhood stats,
and market analytics from the training data.
"""
import os
import pandas as pd
import numpy as np
import config
from models.feature_engineering import load_and_clean_listings, KEY_AMENITIES


_listings_df = None


def _get_listings():
    global _listings_df
    if _listings_df is None:
        try:
            _listings_df = load_and_clean_listings()
        except FileNotFoundError:
            _listings_df = pd.DataFrame()
    return _listings_df


def get_listings_count():
    return len(_get_listings())


def get_neighborhood_stats():
    """Get aggregated stats by neighbourhood_cleansed."""
    df = _get_listings()
    if df.empty or 'neighbourhood_cleansed' not in df.columns:
        return []

    stats = df.groupby('neighbourhood_cleansed').agg(
        count=('price_clean', 'size'),
        avg_price=('price_clean', 'mean'),
        median_price=('price_clean', 'median'),
        avg_occupancy=('occupancy_rate', 'mean'),
        avg_rating=('review_scores_rating', 'mean'),
    ).reset_index()

    stats = stats[stats['count'] >= 5]
    stats['avg_monthly_revenue'] = stats['avg_price'] * stats['avg_occupancy'] * 30

    return stats.sort_values('avg_monthly_revenue', ascending=False).to_dict('records')


def get_comparable_listings(lat, lng, bedrooms, radius_km=3, limit=10):
    """Find comparable Airbnb listings near a location."""
    df = _get_listings()
    if df.empty:
        return []

    # Haversine-ish distance approximation
    dlat = (df['latitude'] - lat) * 111
    dlng = (df['longitude'] - lng) * 85  # rough cos(30°) adjustment for Austin
    df_copy = df.copy()
    df_copy['distance_km'] = np.sqrt(dlat**2 + dlng**2)

    mask = df_copy['distance_km'] <= radius_km
    if bedrooms is not None and 'bedrooms' in df_copy.columns:
        mask = mask & (df_copy['bedrooms'].between(bedrooms - 1, bedrooms + 1))

    comps = df_copy[mask].nsmallest(limit, 'distance_km')

    result = []
    for _, row in comps.iterrows():
        result.append({
            'id': str(row.get('id', '')),
            'name': str(row.get('name', '')),
            'price': round(float(row.get('price_clean', 0)), 2),
            'bedrooms': int(row.get('bedrooms', 0)),
            'bathrooms': round(float(row.get('bathrooms_clean', 1)), 1),
            'occupancy_rate': round(float(row.get('occupancy_rate', 0)), 3),
            'rating': round(float(row.get('review_scores_rating', 0)), 2),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'distance_km': round(float(row['distance_km']), 2),
            'monthly_revenue': round(float(row.get('price_clean', 0)) * float(row.get('occupancy_rate', 0)) * 30, 2),
        })

    return result


def get_market_overview():
    """High-level market stats for analytics page."""
    df = _get_listings()
    if df.empty:
        return {}

    return {
        'total_listings': len(df),
        'avg_nightly_rate': round(float(df['price_clean'].mean()), 2),
        'median_nightly_rate': round(float(df['price_clean'].median()), 2),
        'avg_occupancy': round(float(df['occupancy_rate'].mean()), 4),
        'avg_monthly_revenue': round(float((df['price_clean'] * df['occupancy_rate'] * 30).mean()), 2),
        'avg_rating': round(float(df['review_scores_rating'].mean()), 2),
        'price_distribution': {
            'p10': round(float(df['price_clean'].quantile(0.1)), 2),
            'p25': round(float(df['price_clean'].quantile(0.25)), 2),
            'p50': round(float(df['price_clean'].quantile(0.5)), 2),
            'p75': round(float(df['price_clean'].quantile(0.75)), 2),
            'p90': round(float(df['price_clean'].quantile(0.9)), 2),
        },
        'bedroom_breakdown': _bedroom_breakdown(df),
        'neighborhoods': get_neighborhood_stats()[:15],
    }


def _bedroom_breakdown(df):
    if 'bedrooms' not in df.columns:
        return []
    grouped = df.groupby('bedrooms').agg(
        count=('price_clean', 'size'),
        avg_price=('price_clean', 'mean'),
        avg_occupancy=('occupancy_rate', 'mean'),
    ).reset_index()
    grouped['avg_revenue'] = grouped['avg_price'] * grouped['avg_occupancy'] * 30
    return grouped.to_dict('records')


def get_listing_by_id(listing_id):
    """Look up a single listing by its ID string."""
    df = _get_listings()
    if df.empty or 'id' not in df.columns:
        return None
    match = df[df['id'].astype(str) == str(listing_id)]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        'id': str(row.get('id', '')),
        'name': str(row.get('name', '')),
        'neighbourhood': str(row.get('neighbourhood_cleansed', '')),
        'price': round(float(row.get('price_clean', 0)), 2),
        'bedrooms': int(row.get('bedrooms', 0)),
        'bathrooms': round(float(row.get('bathrooms_clean', 1)), 1),
        'accommodates': int(row.get('accommodates', 0)),
        'occupancy_rate': round(float(row.get('occupancy_rate', 0)), 3),
        'rating': round(float(row.get('review_scores_rating', 0)), 2),
        'latitude': float(row['latitude']),
        'longitude': float(row['longitude']),
        'monthly_revenue': round(float(row.get('price_clean', 0)) * float(row.get('occupancy_rate', 0)) * 30, 2),
    }


def get_all_listings_summary(page=1, per_page=20, min_price=None, max_price=None,
                              bedrooms=None, neighborhood=None, sort_by='price_clean',
                              sort_order='desc'):
    """Return paginated listing summaries for the dashboard (Airbnb data)."""
    df = _get_listings()
    if df.empty:
        return {'items': [], 'total': 0, 'page': page, 'per_page': per_page}

    if min_price is not None:
        df = df[df['price_clean'] >= min_price]
    if max_price is not None:
        df = df[df['price_clean'] <= max_price]
    if bedrooms is not None:
        df = df[df['bedrooms'] == bedrooms]
    if neighborhood and 'neighbourhood_cleansed' in df.columns:
        df = df[df['neighbourhood_cleansed'].str.contains(neighborhood, case=False, na=False)]

    total = len(df)

    ascending = sort_order == 'asc'
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    start = (page - 1) * per_page
    page_df = df.iloc[start:start + per_page]

    items = []
    for _, row in page_df.iterrows():
        items.append({
            'id': str(row.get('id', '')),
            'name': str(row.get('name', '')),
            'neighbourhood': str(row.get('neighbourhood_cleansed', '')),
            'price': round(float(row.get('price_clean', 0)), 2),
            'bedrooms': int(row.get('bedrooms', 0)),
            'bathrooms': round(float(row.get('bathrooms_clean', 1)), 1),
            'accommodates': int(row.get('accommodates', 0)),
            'occupancy_rate': round(float(row.get('occupancy_rate', 0)), 3),
            'rating': round(float(row.get('review_scores_rating', 0)), 2),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'monthly_revenue': round(float(row.get('price_clean', 0)) * float(row.get('occupancy_rate', 0)) * 30, 2),
        })

    return {'items': items, 'total': total, 'page': page, 'per_page': per_page}
