"""The v3 rules and the text/kernel features, without a model fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from evaluation.intervals import (
    apply_log_radii,
    conformal_radius,
    cqr_bounds,
    cqr_nonconformity,
    fit_log_scale,
    sigma_of,
)
from evaluation.select import choose_interval, choose_pooled, choose_sequential
from features.columns import DETAIL_COLUMNS
from features.fold import KernelSmoother
from features.frame import attach_base_features
from features.text import FoldText, lexicon_frame


def test_lexicon_flags_and_empty_text_is_missing():
    frame = lexicon_frame(
        [
            "Luxury loft with a skyline view",
            "   ",
            "A quiet cottage. Walk to downtown and the bay.",
        ]
    )
    assert frame.loc[0, "text_mentions_luxury"] == 1.0
    assert frame.loc[0, "text_mentions_view"] == 1.0
    assert np.isnan(frame.loc[1, "log_text_chars"])
    assert frame.loc[2, "text_mentions_quiet"] == 1.0
    assert frame.loc[2, "text_mentions_walk"] == 1.0
    assert frame.loc[2, "text_mentions_water"] == 1.0
    # "review" is not a view, and "new" by itself is not new construction.
    plain = lexicon_frame(["A great review of a new coffee shop"])
    assert plain.loc[0, "text_mentions_view"] == 0.0
    assert plain.loc[0, "text_mentions_new"] == 0.0


def test_value_score_is_not_a_feature():
    # Guests judge value against the price. That column is a consequence of
    # the label, so it is not eligible.
    assert not any("value" in name for name in DETAIL_COLUMNS)


def test_text_encoder_does_not_need_the_held_out_document():
    train = np.array(["alpha alpha luxury view", "beta beta quiet room", "alpha beta downtown"])
    encoder = FoldText(n_components=2, min_df=1).fit(train)
    first = encoder.transform(np.array(["alpha alpha luxury view"]))
    again = (
        FoldText(n_components=2, min_df=1)
        .fit(train)
        .transform(np.array(["alpha alpha luxury view"]))
    )
    np.testing.assert_allclose(first, again)
    held_out = encoder.transform(np.array(["zzzzzz unique token that was not in training"]))
    assert held_out.shape == (1, 2)
    assert np.isfinite(held_out).all()


def test_kernel_does_not_use_a_rows_own_price():
    lat = np.array([30.0, 30.0, 30.0])
    lon = np.array([-97.0, -97.01, -97.02])
    log_price = np.log1p(np.array([100.0, 110.0, 900.0]))
    smoother = KernelSmoother(bandwidth_km=50.0, n_neighbors=2).fit(lat, lon, log_price)
    values = smoother.transform(lat, lon, row_index=np.arange(3))
    # The $900 listing can see the $100 and $110 listings, not itself.
    assert values[2] < np.log1p(200)


def test_shared_bath_and_host_tenure_from_the_listing_fields():
    frame = pd.DataFrame(
        {
            "latitude": [30.27],
            "longitude": [-97.74],
            "bedrooms": [1],
            "bathrooms_clean": [1.0],
            "accommodates": [2],
            "property_type": ["Apartment"],
            "room_type": ["Entire home/apt"],
            "minimum_nights": [2],
            "number_of_reviews": [0],
            "reviews_per_month": [np.nan],
            "review_scores_rating": [np.nan],
            "amenities": ['["Fireplace", "Dishwasher"]'],
            "host_is_superhost": ["f"],
            "host_listings_count": [1],
            "instant_bookable": ["t"],
            "bathrooms_text": ["1 shared bath"],
            "beds": [1],
            "host_since": ["2020-06-13"],
            "host_response_rate": ["80%"],
            "host_acceptance_rate": ["N/A"],
            "host_response_time": ["within an hour"],
            "host_identity_verified": ["t"],
            "name": ["Quiet studio"],
            "description": ["A quiet room."],
            "neighborhood_overview": [""],
        }
    )
    out = attach_base_features(frame, "austin")
    assert out.loc[0, "bath_is_shared"] == 1.0
    assert out.loc[0, "has_fireplace"] == 1.0
    assert out.loc[0, "has_dishwasher"] == 1.0
    assert out.loc[0, "host_response_rate"] == 0.8
    assert np.isnan(out.loc[0, "host_acceptance_rate"])
    assert out.loc[0, "host_response_time_ord"] == 1.0
    # Host since 2020-06-13 is about five years before the Austin snapshot.
    assert 1.7 < out.loc[0, "log_host_tenure"] < 1.9
    assert out.loc[0, "text_mentions_quiet"] == 1.0


def test_sequential_addon_rule():
    # 99.2 is not a 1% win on 100. 90 is. 89.5 is not a 1% win on 90.
    kept = choose_sequential(100.0, [("detail", 99.2), ("lexicon", 90.0), ("svd", 89.5)])
    assert kept == ["lexicon"]
    assert choose_sequential(100.0, [("detail", 90.0), ("lexicon", 80.0)]) == [
        "detail",
        "lexicon",
    ]
    assert choose_pooled(100.0, 98.5) is True
    assert choose_pooled(100.0, 99.5) is False


def test_interval_rule_keeps_the_band_unless_the_tail_improves():
    baseline = {
        "name": "log_conformal",
        "mean_coverage": 0.80,
        "worst_coverage": 0.79,
        "worst_upper_coverage": 0.70,
        "mean_median_width": 100.0,
        "point_inside": 1.0,
    }
    wider = {
        "name": "scaled_log",
        "mean_coverage": 0.80,
        "worst_coverage": 0.79,
        "worst_upper_coverage": 0.74,
        "mean_median_width": 160.0,
        "point_inside": 1.0,
    }
    better_tail = {
        "name": "cqr",
        "mean_coverage": 0.81,
        "worst_coverage": 0.79,
        "worst_upper_coverage": 0.75,
        "mean_median_width": 110.0,
        "point_inside": 0.95,
    }
    misses_the_point = {**better_tail, "name": "cqr", "point_inside": 0.5}
    assert choose_interval([baseline, wider]) == "log_conformal"
    assert choose_interval([baseline, better_tail]) == "cqr"
    assert choose_interval([baseline, misses_the_point]) == "log_conformal"


def test_plain_reason_names_the_fact():
    from models.explain import plain_reason

    studio = plain_reason("bedrooms", 0, -12)
    assert "studio" in studio
    assert "down" in studio
    view = plain_reason("text_mentions_view", 1, 6)
    assert "view" in view
    assert "up" in view


def test_scaled_interval_and_cqr_scores():
    pred = np.array([100.0, 100.0, 400.0, 400.0])
    residual = np.array([0.10, 0.10, 0.30, 0.30])
    intercept, slope = fit_log_scale(pred, residual)
    assert slope > 0
    sigma = sigma_of(pred, intercept, slope, floor=0.05)
    assert sigma[2] > sigma[0]
    low, high = apply_log_radii(pred, np.array([0.2, 0.2, 0.2, 0.2]))
    assert high[2] - low[2] > high[0] - low[0]

    inside = cqr_nonconformity(np.expm1(np.array([1.0])), np.array([0.5]), np.array([1.0]))
    assert abs(inside[0]) < 1e-9
    outside = cqr_nonconformity(np.expm1(np.array([1.2])), np.array([0.5]), np.array([1.0]))
    assert abs(outside[0] - 0.2) < 1e-9
    low_b, high_b = cqr_bounds(np.array([0.5]), np.array([1.0]), conformal_radius(np.array([0.2])))
    assert low_b[0] < high_b[0]
