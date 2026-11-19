"""v4 rules and leak checks. No model is fit here."""

from __future__ import annotations

import numpy as np
import pandas as pd
from data.history import dedupe_snapshots
from evaluation.intervals import (
    apply_mondrian,
    apply_signed_interval,
    asymmetric_quantiles,
    fit_mondrian,
)
from evaluation.select import choose_band_interval, choose_confirmed, choose_correction
from evaluation.v4 import apply_smear, filter_extra, smear_factor, tail_weights
from features.comps import MatchedComps
from features.embed import FoldEmbed


def test_tail_weight_is_one_at_the_median_and_three_at_the_90th():
    prices = np.array([50, 100, 100, 100, 400], dtype=float)
    weights = tail_weights(prices)
    assert weights.min() >= 1.0 - 1e-9
    assert weights.max() <= 3.0 + 1e-9
    # The median log price has weight 1. The top of this sample is the cap.
    median_price = float(np.expm1(np.quantile(np.log1p(prices), 0.50)))
    at_median = tail_weights(np.array([median_price]))
    assert abs(at_median[0] - 1.0) < 1e-6


def test_matched_comps_do_not_use_a_rows_own_price():
    lat = np.array([30.0, 30.0, 30.01, 30.2])
    lon = np.array([-97.0, -97.01, -97.02, -97.2])
    rooms = np.array(["Entire home/apt"] * 4)
    beds = np.array([2, 2, 2, 2], dtype=float)
    log_price = np.log1p(np.array([100.0, 110.0, 120.0, 900.0]))
    comps = MatchedComps(max_km=50.0, min_matches=1).fit(lat, lon, log_price, rooms, beds)
    values = comps.transform(lat, lon, rooms, beds, row_index=np.arange(4))
    # The $900 listing can see the cheap ones. It cannot see itself.
    assert values[3] < np.log1p(200)


def test_embedding_pca_does_not_need_the_held_out_sentence():
    def encoder(texts):
        # Two fixed axes, so the test does not download a model.
        rows = []
        for text in texts:
            rows.append([float(len(text)), float(text.count("luxury"))])
        return np.array(rows, dtype=float)

    train = ["a quiet room", "luxury loft with a view", "small studio"]
    model = FoldEmbed(n_components=2, encoder=encoder).fit(train)
    first = model.transform(["luxury loft with a view"])
    again = (
        FoldEmbed(n_components=2, encoder=encoder).fit(train).transform(["luxury loft with a view"])
    )
    np.testing.assert_allclose(first, again)
    held_out = model.transform(["zzzz something new"])
    assert held_out.shape == (1, 2)
    assert np.isfinite(held_out).all()
    empty = model.transform(["   "])
    assert np.isnan(empty).all()


def test_history_dedupe_keeps_the_newest_and_drops_study_ids():
    older = pd.DataFrame({"id": [1, 2], "price_clean": [80.0, 90.0], "months_before": [12.0, 12.0]})
    newer = pd.DataFrame({"id": [2, 3], "price_clean": [110.0, 130.0], "months_before": [3.0, 3.0]})
    # Newest first, which is the order load_history passes in.
    extra = dedupe_snapshots([newer, older], study_ids={"1"})
    ids = set(extra["id"].astype(int))
    assert ids == {2, 3}
    kept = extra.loc[extra["id"] == 2, "price_clean"].iloc[0]
    assert kept == 110.0


def test_filter_extra_drops_a_held_out_cell_and_a_shared_host():
    frame = pd.DataFrame(
        {
            "latitude": [30.00, 30.50],
            "longitude": [-97.00, -97.50],
            "host_id": ["a", "b"],
        }
    )
    extra = pd.DataFrame(
        {
            "latitude": [30.00, 31.00, 32.00],
            "longitude": [-97.00, -98.00, -99.00],
            "host_id": ["z", "a", "q"],
            "id": [10, 11, 12],
        }
    )
    # Row 0 is held out. The extra in that cell goes, and so does the extra
    # whose host has a held-out listing. The far listing stays.
    kept = filter_extra(extra, frame, eval_idx=np.array([0]), blocked_idx=np.array([], dtype=int))
    assert set(kept["id"].astype(int)) == {12}


def test_smear_raises_a_low_prediction_toward_the_mean():
    y_true = np.array([100.0, 100.0, 400.0])
    pred = np.array([100.0, 100.0, 100.0])
    factor = smear_factor(y_true, pred)
    assert factor > 1.0
    adjusted = apply_smear(pred, factor)
    assert adjusted.mean() > pred.mean()


def test_asymmetric_interval_gives_more_room_above_a_low_model():
    y_true = np.array([100, 120, 150, 180, 220, 260, 300, 400], dtype=float)
    pred = np.full(len(y_true), 120.0)
    low_q, high_q = asymmetric_quantiles(np.log1p(y_true) - np.log1p(pred), level=0.5)
    assert high_q > 0
    assert high_q > abs(low_q)
    low, high = apply_signed_interval(pred, low_q, high_q)
    assert np.all(high >= low)
    covered = ((y_true >= low) & (y_true <= high)).mean()
    assert covered >= 0.5 - 1e-9


def test_mondrian_bands_are_not_one_radius():
    rng = np.random.default_rng(0)
    pred = np.concatenate([np.full(40, 80.0), np.full(40, 400.0)])
    # Cheap predictions are noisy. Expensive ones are not.
    cheap = pred[:40] * np.exp(rng.normal(0, 0.4, 40))
    rich = pred[40:] * np.exp(rng.normal(0, 0.05, 40))
    y_true = np.concatenate([cheap, rich])
    residual = np.log1p(y_true) - np.log1p(pred)
    params = fit_mondrian(pred, residual, n_bands=2, min_rows=20, level=0.8)
    cheap_span = params["bands"][0][1] - params["bands"][0][0]
    rich_span = params["bands"][1][1] - params["bands"][1][0]
    assert cheap_span > rich_span
    low, high = apply_mondrian(pred, params)
    assert np.all(high >= low)


def test_correction_and_band_rules_are_the_ones_the_report_replays():
    assert choose_correction(100.0, [("smear", 99.5), ("isotonic", 99.2)]) == "none"
    assert choose_correction(100.0, [("smear", 98.0), ("isotonic", 97.5)]) == "smear"
    assert choose_correction(100.0, [("smear", 98.0), ("isotonic", 90.0)]) == "isotonic"
    assert choose_confirmed(100.0, 98.5) is True
    assert choose_confirmed(100.0, 99.5) is False
    incumbent = {
        "name": "scaled_log",
        "mean_coverage": 0.80,
        "point_inside": 0.99,
        "worst_coverage": 0.79,
        "mean_median_width": 120.0,
        "worst_upper_coverage": 0.75,
        "worst_band_coverage": 0.60,
    }
    short = {
        **incumbent,
        "name": "asymmetric",
        "worst_band_coverage": 0.62,
    }
    fixed = {
        **incumbent,
        "name": "mondrian_asymmetric",
        "worst_band_coverage": 0.70,
        "mean_median_width": 130.0,
    }
    assert choose_band_interval([incumbent, short, fixed]) == "mondrian_asymmetric"
    too_wide = {**fixed, "mean_median_width": 200.0}
    assert choose_band_interval([incumbent, too_wide]) == "scaled_log"
