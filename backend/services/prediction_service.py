import os
import calendar
import joblib
import numpy as np
import pandas as pd
import config
from cities import DEFAULT_CITY, get_seasonal_factors
from models.feature_engineering import get_feature_columns, KEY_AMENITIES, haversine_km


_models_cache = {}


def _load_models(city_key=None):
    city_key = city_key or DEFAULT_CITY

    if city_key in _models_cache:
        return _models_cache[city_key]

    models_dir = config.city_models_dir(city_key)
    if not os.path.exists(os.path.join(models_dir, 'price_model.joblib')):
        _models_cache[city_key] = {}
        return _models_cache[city_key]

    entry = {}
    for name in ['price_model', 'occupancy_model', 'feature_columns',
                  'feature_medians', 'feature_stats',
                  'monthly_feature_columns', 'monthly_feature_medians']:
        path = os.path.join(models_dir, f'{name}.joblib')
        if os.path.exists(path):
            entry[name] = joblib.load(path)

    _models_cache[city_key] = entry
    return entry


def models_available(city_key=None):
    m = _load_models(city_key)
    return 'price_model' in m and 'occupancy_model' in m


def _is_monthly_model(m):
    """Check if models were trained with month features."""
    cols = m.get('monthly_feature_columns', m.get('occupancy_feature_columns'))
    return cols is not None and 'sin_month' in cols


def _get_monthly_cols(m):
    """Get the monthly feature columns (new or legacy name)."""
    return m.get('monthly_feature_columns', m.get('occupancy_feature_columns'))


def _get_monthly_medians(m):
    """Get the monthly feature medians (new or legacy name)."""
    return m.get('monthly_feature_medians',
                 m.get('occupancy_feature_medians',
                        m.get('feature_medians', {})))


def _build_base_row(features_dict, m, city_key=None):
    """Build a feature row using base feature columns with median defaults."""
    feature_cols = m['feature_columns']
    medians = m.get('feature_medians', {})

    row = {}
    for col in feature_cols:
        if col in features_dict:
            row[col] = features_dict[col]
        elif col in medians:
            row[col] = medians[col]
        else:
            row[col] = 0

    if 'accommodates' not in features_dict and 'bedrooms' in features_dict:
        row['accommodates'] = features_dict['bedrooms'] * 2

    # Compute distance_to_center if not provided
    if 'distance_to_center' not in features_dict and 'latitude' in features_dict:
        from cities import get_city_or_default
        city = get_city_or_default(city_key)
        row['distance_to_center'] = float(haversine_km(
            features_dict['latitude'], features_dict.get('longitude', city['longitude']),
            city['latitude'], city['longitude'],
        ))

    return row


def _build_monthly_row(base_row, month, m):
    """Extend a base feature row with month encoding for monthly models."""
    row = dict(base_row)
    row['sin_month'] = np.sin(2 * np.pi * month / 12)
    row['cos_month'] = np.cos(2 * np.pi * month / 12)

    monthly_cols = _get_monthly_cols(m)
    medians = _get_monthly_medians(m)
    for col in monthly_cols:
        if col not in row:
            row[col] = medians.get(col, 0)

    return row


def _predict_for_month(base_row, month, m):
    """Predict both price and occupancy for a single month."""
    row = _build_monthly_row(base_row, month, m)
    monthly_cols = _get_monthly_cols(m)
    X = pd.DataFrame([row], columns=monthly_cols)

    nightly_rate = float(m['price_model'].predict(X)[0])
    occupancy = float(np.clip(m['occupancy_model'].predict(X)[0], 0, 1))

    return nightly_rate, occupancy


def _predict_all_months(base_row, m):
    """Predict price and occupancy for all 12 months in a single batch."""
    monthly_cols = _get_monthly_cols(m)
    rows = []
    for mo in range(1, 13):
        rows.append(_build_monthly_row(base_row, mo, m))
    X = pd.DataFrame(rows, columns=monthly_cols)

    rates = m['price_model'].predict(X).tolist()
    occs = np.clip(m['occupancy_model'].predict(X), 0, 1).tolist()
    return rates, occs


def _compute_confidence(base_row, m):
    """
    Score prediction confidence primarily on geographic proximity to
    training data — how many Airbnb listings the model actually learned
    from in this property's neighborhood.

    Structural features (bedrooms, bathrooms) are universally applicable
    and get minimal weight. Location is what matters.
    """
    stats = m.get('feature_stats')
    if not stats:
        return {'level': 'medium', 'score': 50, 'nearby_listings': 0,
                'radius_km': 3, 'n_training': 0, 'out_of_range_features': []}

    target_stats = stats.get('targets', {})
    n_training = target_stats.get('n_training', 0)
    coords = stats.get('training_coords')

    lat = base_row.get('latitude', 0)
    lng = base_row.get('longitude', 0)

    # --- Geographic density: count training listings within radius ---
    nearby_1km = 0
    nearby_3km = 0
    nearby_5km = 0

    if coords is not None and lat and lng:
        train_lat = coords['latitude']
        train_lng = coords['longitude']
        # Vectorized haversine (fast for <20k points)
        dists = haversine_km(lat, lng, train_lat, train_lng)
        nearby_1km = int(np.sum(dists <= 1.0))
        nearby_3km = int(np.sum(dists <= 3.0))
        nearby_5km = int(np.sum(dists <= 5.0))

    # --- Score: primarily driven by nearby listing density ---
    # 50+ within 3km = excellent local data → high confidence
    # 15-49 = decent coverage → medium
    # <15 = sparse → low
    if nearby_3km >= 50:
        geo_score = 90 + min(10, (nearby_3km - 50) // 10)  # 90-100
    elif nearby_3km >= 15:
        geo_score = 50 + round((nearby_3km - 15) / 35 * 40)  # 50-90
    else:
        geo_score = max(5, round(nearby_3km / 15 * 50))  # 0-50

    score = min(100, geo_score)

    if score >= 70:
        level = 'high'
    elif score >= 40:
        level = 'medium'
    else:
        level = 'low'

    # --- Flag distance_to_center if extreme (property is far from city) ---
    feat_stats = stats.get('features', {})
    out_of_range = []
    dist = base_row.get('distance_to_center', 0)
    if 'distance_to_center' in feat_stats:
        fs = feat_stats['distance_to_center']
        if dist > fs['p95']:
            out_of_range.append({
                'feature': 'distance_to_center',
                'value': round(dist, 1),
                'typical_range': [round(fs['p5'], 1), round(fs['p95'], 1)],
                'detail': f'{round(dist, 1)} km from city center (most listings are within {round(fs["p95"], 1)} km)',
            })

    return {
        'level': level,
        'score': score,
        'nearby_1km': nearby_1km,
        'nearby_3km': nearby_3km,
        'nearby_5km': nearby_5km,
        'radius_km': 3,
        'n_training': n_training,
        'out_of_range_features': out_of_range,
    }


def predict_property(features_dict, month=None, city_key=None, summary_only=False):
    """
    Predict nightly rate and occupancy for a property.

    features_dict should contain: bedrooms, bathrooms_clean, accommodates, latitude, longitude
    Optional: amenity_count, beds, has_* amenity flags, is_* property type flags,
              sqft, year_built, lot_sqft, property_age, sqft_per_bedroom, lot_acres

    month: optional int (1-12). If given, returns prediction for that month.
    If None, returns full 12-month breakdown with seasonal adjustments.
    summary_only: if True, skip monthly breakdown and confidence (faster for list scoring).
    """
    m = _load_models(city_key)
    if 'price_model' not in m or 'occupancy_model' not in m:
        raise RuntimeError("Models not trained. Run train_model.py first.")
    if 'feature_columns' not in m or 'feature_medians' not in m:
        raise RuntimeError("Feature metadata not found. Run train_model.py first.")

    base_row = _build_base_row(features_dict, m, city_key)
    feature_cols = m['feature_columns']
    has_monthly = _is_monthly_model(m)

    # --- Step 1: Get per-month predictions ---
    if has_monthly:
        # Batched: predict all 12 months in 2 model calls (not 24)
        monthly_rates, raw_monthly_occs = _predict_all_months(base_row, m)

        # Light guardrails: floor at 5% (avoid exact zeros from model noise)
        # and cap at 95%. No annual scaling — trust the model's output.
        monthly_occs = [float(np.clip(occ, 0.05, 0.95))
                        for occ in raw_monthly_occs]
    else:
        # No monthly model: flat base prediction + city seasonal factors
        X = pd.DataFrame([base_row], columns=feature_cols)
        base_rate = float(m['price_model'].predict(X)[0])
        base_occ = float(np.clip(m['occupancy_model'].predict(X)[0], 0, 1))
        seasonal = get_seasonal_factors(city_key)
        monthly_rates = [base_rate * seasonal['price'][mo] for mo in range(12)]
        monthly_occs = [float(np.clip(base_occ * seasonal['occupancy'][mo], 0, 1)) for mo in range(12)]

    # --- Step 2: Compute annual totals ---
    total_revenue = 0.0
    for mo in range(12):
        days = calendar.monthrange(2025, mo + 1)[1]
        total_revenue += monthly_rates[mo] * monthly_occs[mo] * days

    nightly_rate = sum(monthly_rates) / 12
    occupancy_rate = sum(monthly_occs) / 12
    annual_revenue = total_revenue
    monthly_revenue = annual_revenue / 12

    # Fast path for list scoring — skip breakdown but keep confidence
    if summary_only:
        confidence = _compute_confidence(base_row, m)
        return {
            'predicted_nightly_rate': round(nightly_rate, 2),
            'predicted_occupancy_rate': round(occupancy_rate, 4),
            'predicted_monthly_revenue': round(monthly_revenue, 2),
            'predicted_annual_revenue': round(annual_revenue, 2),
            'monthly_breakdown': None,
            'confidence': confidence,
            'prediction_range': None,
        }

    # --- Step 3: Build monthly breakdown ---
    if month is not None:
        mr = monthly_rates[month - 1]
        mo_occ = monthly_occs[month - 1]
        days = calendar.monthrange(2025, month)[1]
        monthly_revenue = mr * mo_occ * days
        annual_revenue = monthly_revenue * 12
        nightly_rate = mr
        occupancy_rate = mo_occ
        monthly_breakdown = None
    else:
        monthly_breakdown = []
        for mo in range(1, 13):
            rate = monthly_rates[mo - 1]
            occ = monthly_occs[mo - 1]
            days = calendar.monthrange(2025, mo)[1]
            rev = rate * occ * days
            monthly_breakdown.append({
                'month': mo,
                'nightly_rate': round(rate, 2),
                'occupancy_rate': round(occ, 4),
                'monthly_revenue': round(rev, 2),
            })

    # --- Step 3: Confidence scoring & prediction range ---
    confidence = _compute_confidence(base_row, m)
    range_mult = {'high': 0.15, 'medium': 0.30, 'low': 0.50}[confidence['level']]

    result = {
        'predicted_nightly_rate': round(nightly_rate, 2),
        'predicted_occupancy_rate': round(occupancy_rate, 4),
        'predicted_monthly_revenue': round(monthly_revenue, 2),
        'predicted_annual_revenue': round(annual_revenue, 2),
        'monthly_breakdown': monthly_breakdown,
        'confidence': confidence,
        'prediction_range': {
            'nightly_low': round(nightly_rate * (1 - range_mult), 2),
            'nightly_high': round(nightly_rate * (1 + range_mult), 2),
            'annual_revenue_low': round(annual_revenue * (1 - range_mult), 2),
            'annual_revenue_high': round(annual_revenue * (1 + range_mult), 2),
            'occupancy_low': round(float(np.clip(occupancy_rate * (1 - range_mult), 0, 1)), 4),
            'occupancy_high': round(float(np.clip(occupancy_rate * (1 + range_mult), 0, 1)), 4),
        },
    }

    return result
