import os
import joblib
import numpy as np
import pandas as pd
import config
from models.feature_engineering import get_feature_columns, KEY_AMENITIES


_price_model = None
_occ_model = None
_feature_cols = None
_medians = None


def _load_models():
    global _price_model, _occ_model, _feature_cols, _medians

    models_dir = config.TRAINED_MODELS_DIR

    price_path = os.path.join(models_dir, 'price_model.joblib')
    occ_path = os.path.join(models_dir, 'occupancy_model.joblib')
    cols_path = os.path.join(models_dir, 'feature_columns.joblib')
    medians_path = os.path.join(models_dir, 'feature_medians.joblib')

    if os.path.exists(price_path):
        _price_model = joblib.load(price_path)
    if os.path.exists(occ_path):
        _occ_model = joblib.load(occ_path)
    if os.path.exists(cols_path):
        _feature_cols = joblib.load(cols_path)
    if os.path.exists(medians_path):
        _medians = joblib.load(medians_path)


def get_models():
    if _price_model is None:
        _load_models()
    return _price_model, _occ_model


def models_available():
    price_model, occ_model = get_models()
    return price_model is not None and occ_model is not None


def predict_property(features_dict):
    """
    Predict nightly rate and occupancy for a property.

    features_dict should contain: bedrooms, bathrooms, accommodates, latitude, longitude
    Optional: amenity_count, beds, review_scores_*, has_* amenity flags
    """
    price_model, occ_model = get_models()
    if price_model is None or occ_model is None:
        raise RuntimeError("Models not trained. Run train_model.py first.")

    if _feature_cols is None or _medians is None:
        raise RuntimeError("Feature metadata not found. Run train_model.py first.")

    # Build feature vector with defaults from training medians
    row = {}
    for col in _feature_cols:
        if col in features_dict:
            row[col] = features_dict[col]
        elif col in _medians:
            row[col] = _medians[col]
        else:
            row[col] = 0

    # Default accommodates to bedrooms * 2 if not provided
    if 'accommodates' not in features_dict and 'bedrooms' in features_dict:
        row['accommodates'] = features_dict['bedrooms'] * 2

    X = pd.DataFrame([row], columns=_feature_cols)

    nightly_rate = float(price_model.predict(X)[0])
    occupancy_rate = float(np.clip(occ_model.predict(X)[0], 0, 1))

    return {
        'predicted_nightly_rate': round(nightly_rate, 2),
        'predicted_occupancy_rate': round(occupancy_rate, 4),
        'predicted_monthly_revenue': round(nightly_rate * occupancy_rate * 30, 2),
        'predicted_annual_revenue': round(nightly_rate * occupancy_rate * 365, 2),
    }
