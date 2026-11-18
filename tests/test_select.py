"""The selection rules are pure functions. The report test replays them on the JSON."""

import numpy as np
from evaluation.select import (
    choose_addon,
    choose_blend,
    choose_depth,
    choose_library,
    choose_monotonic,
    choose_quality,
    choose_simpler,
    feature_label,
)


def test_simpler_prefix_wins_a_one_percent_tie():
    # The later prefix is best, but the earlier one is within 1%.
    assert choose_simpler([("size", 100.0), ("location", 99.5), ("census", 99.2)]) == "size"


def test_addon_must_earn_a_percent_and_smooth_wins_a_tie():
    assert choose_addon(100.0, [("target_smooth", 99.5), ("target_knn", 99.2)]) is None
    assert choose_addon(100.0, [("target_smooth", 98.9), ("target_knn", 98.5)]) == "target_smooth"
    assert choose_addon(100.0, [("target_smooth", 99.5), ("target_knn", 97.0)]) == "target_knn"


def test_quality_library_depth_blend_rules():
    assert choose_quality(100.0, 98.9) is True
    assert choose_quality(100.0, 99.2) is False
    assert choose_depth([(8, 50.0), (4, 50.4), (6, 50.2)]) == 4
    # 1% better than XGBoost is not enough. 2% is, and the better of those wins.
    assert choose_library({"xgboost": 100.0, "lightgbm": 99.0, "catboost": 97.5}) == "catboost"
    assert choose_library({"xgboost": 100.0, "lightgbm": 99.0, "catboost": 98.5}) == "xgboost"
    assert choose_monotonic(100.0, 100.5) is True
    assert choose_monotonic(100.0, 102.0) is False
    assert choose_blend(100.0, 98.9) is True
    assert choose_blend(100.0, 99.5) is False
    assert feature_label("plus_osm", "target_smooth", False) == "plus_osm+target_smooth"
    assert feature_label("plus_location", None, False) == "plus_location"


def test_conformal_radius_covers_a_fixed_sample():
    from evaluation.intervals import apply_log_interval, conformal_radius

    residual = np.array([0.1, 0.2, 0.3, 0.4, 1.0])
    radius = conformal_radius(residual, level=0.8)
    # The finite-sample quantile is at least the empirical 80% and finite.
    assert 0.4 <= radius <= 1.0
    low, high = apply_log_interval(np.array([100.0]), radius)
    assert low[0] < 100.0 < high[0]
    # A log radius is multiplicative: a $400 night gets a wider dollar band.
    low_hi, high_hi = apply_log_interval(np.array([400.0]), radius)
    assert (high_hi[0] - low_hi[0]) > (high[0] - low[0])
