"""Local scoring. The study model predicts one nightly rate; the monthly
curve is the hand-set seasonal weights in data/cities.py, which average to
1 so they do not change the annual mean. Occupancy is the neighbourhood
median of the review-based proxy, not a second trained model.
"""

import calendar
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.artifacts import available, load_artifact
from models.score import score_features

import config
from cities import DEFAULT_CITY, get_seasonal_factors

_cache = {}
_trees = {}
EARTH_RADIUS_KM = 6371.0


def _load(city_key=None):
    city_key = city_key or DEFAULT_CITY
    if city_key not in _cache:
        _cache[city_key] = load_artifact(city_key) if available(city_key) else None
    return _cache[city_key]


def models_available(city_key=None):
    city_key = city_key or DEFAULT_CITY
    if city_key in _cache:
        return _cache[city_key] is not None
    return available(city_key)


def get_models(city_key=None):
    artifact = _load(city_key)
    if not artifact:
        return {}
    return {"price_model": artifact["model"]}


def _balltree(city_key):
    if city_key in _trees:
        return _trees[city_key]
    artifact = _load(city_key)
    if not artifact:
        _trees[city_key] = None
        return None
    lat = np.asarray(artifact["training_latitude"])
    lon = np.asarray(artifact["training_longitude"])
    _trees[city_key] = BallTree(np.deg2rad(np.column_stack([lat, lon])), metric="haversine")
    return _trees[city_key]


def _confidence(lat, lon, city_key, artifact):
    tree = _balltree(city_key)
    nearby = 0
    if tree is not None and lat is not None and lon is not None:
        point = np.deg2rad([[lat, lon]])
        nearby = int(tree.query_radius(point, r=3.0 / EARTH_RADIUS_KM, count_only=True)[0])
    if nearby >= 80:
        level, score = "high", 85
    elif nearby >= 25:
        level, score = "medium", 55
    else:
        level, score = "low", 25
    out_of_range = []
    return {
        "level": level,
        "score": score,
        "nearby_3km": nearby,
        "radius_km": 3,
        "n_training": int(len(artifact["training_latitude"])),
        "out_of_range_features": out_of_range,
    }


def predict_property(features_dict, month=None, city_key=None, summary_only=False):
    city_key = city_key or DEFAULT_CITY
    artifact = _load(city_key)
    if artifact is None:
        raise RuntimeError(
            "No model for this city. From the repo root: python -m evaluation.run_eval"
        )

    scored = score_features(features_dict, city_key)
    base_rate = scored["nightly_rate"]
    base_occ = float(np.clip(scored["occupancy_rate"], 0.05, 0.70))
    seasonal = get_seasonal_factors(city_key)
    monthly_rates = [base_rate * seasonal["price"][mo] for mo in range(12)]
    monthly_occs = [
        float(np.clip(base_occ * seasonal["occupancy"][mo], 0.0, 0.70)) for mo in range(12)
    ]

    total_revenue = 0.0
    for mo in range(12):
        days = calendar.monthrange(2026, mo + 1)[1]
        total_revenue += monthly_rates[mo] * monthly_occs[mo] * days

    if month is not None:
        nightly_rate = monthly_rates[month - 1]
        occupancy_rate = monthly_occs[month - 1]
        days = calendar.monthrange(2026, month)[1]
        monthly_revenue = nightly_rate * occupancy_rate * days
        annual_revenue = monthly_revenue * 12
        monthly_breakdown = None
    else:
        nightly_rate = sum(monthly_rates) / 12
        occupancy_rate = sum(monthly_occs) / 12
        annual_revenue = total_revenue
        monthly_revenue = annual_revenue / 12
        monthly_breakdown = []
        for mo in range(1, 13):
            days = calendar.monthrange(2026, mo)[1]
            monthly_breakdown.append(
                {
                    "month": mo,
                    "nightly_rate": round(monthly_rates[mo - 1], 2),
                    "occupancy_rate": round(monthly_occs[mo - 1], 4),
                    "monthly_revenue": round(
                        monthly_rates[mo - 1] * monthly_occs[mo - 1] * days, 2
                    ),
                }
            )

    lat = features_dict.get("latitude")
    lon = features_dict.get("longitude")
    confidence = _confidence(lat, lon, city_key, artifact)
    result = {
        "predicted_nightly_rate": round(nightly_rate, 2),
        "predicted_occupancy_rate": round(occupancy_rate, 4),
        "predicted_monthly_revenue": round(monthly_revenue, 2),
        "predicted_annual_revenue": round(annual_revenue, 2),
        "monthly_breakdown": None if summary_only else monthly_breakdown,
        "confidence": confidence,
        "occupancy_source": scored["occupancy_source"],
        "model_feature_set": scored["feature_set"],
        "prediction_range": None,
    }
    if not summary_only:
        span = {"high": 0.15, "medium": 0.30, "low": 0.50}[confidence["level"]]
        result["prediction_range"] = {
            "nightly_low": round(nightly_rate * (1 - span), 2),
            "nightly_high": round(nightly_rate * (1 + span), 2),
            "annual_revenue_low": round(annual_revenue * (1 - span), 2),
            "annual_revenue_high": round(annual_revenue * (1 + span), 2),
            "occupancy_low": round(float(np.clip(occupancy_rate * (1 - span), 0, 1)), 4),
            "occupancy_high": round(float(np.clip(occupancy_rate * (1 + span), 0, 1)), 4),
        }
    return result


# Kept so older imports do not crash. The study no longer writes this path.
def city_has_legacy_files(city_key=None):
    city_key = city_key or DEFAULT_CITY
    return os.path.isfile(os.path.join(config.city_models_dir(city_key), "price_model.joblib"))
