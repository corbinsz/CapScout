"""The nights label, the grade, and the frozen occupancy holdout.

The holdout file is listing ids only. The report tests recompute the
holdout metrics from the saved predictions. They also replay the pre-set
bar, so a later edit cannot flip the cleared flag without changing the rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from data.occupancy import ANNUAL_NIGHTS_CAP, booked_nights_ltm
from evaluation.audit import split_audit
from evaluation.freeze import listing_ids
from evaluation.metrics import regression_metrics
from evaluation.occupancy_freeze import PATH, computed_ids, occupancy_split, population
from evaluation.occupancy_protocol import (
    FEATURE_LADDER,
    FORBIDDEN_COLUMNS,
    OCCUPANCY_SEED,
    assert_columns_allowed,
    cleared_classification,
    cleared_regression,
)
from models.grade import annual_revenue, grade_revenue, gross_yield

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "occupancy_metrics.json"
PREDICTIONS = ROOT / "reports" / "occupancy_predictions.csv"
OLD_SEEDS = {0, 1, 2, 3, 4, 7, 11, 23, 42}


def test_booked_nights_follow_the_stated_rule():
    # 1 review, 50% review rate, 3-night stay -> 6 nights.
    nights = booked_nights_ltm(np.array([1.0]), np.array([1.0]), 3.0)
    assert abs(nights[0] - 6.0) < 1e-9

    # A longer minimum replaces the 3-night stay.
    longer = booked_nights_ltm(np.array([1.0]), np.array([7.0]), 3.0)
    assert abs(longer[0] - 14.0) < 1e-9

    # A 120-night minimum is not a stay length. Fall back to 3 nights.
    ignored = booked_nights_ltm(np.array([1.0]), np.array([120.0]), 3.0)
    assert abs(ignored[0] - 6.0) < 1e-9

    # The annual cap is 70% of 365, which is 255.5 nights.
    capped = booked_nights_ltm(np.array([100.0]), np.array([1.0]), 3.0)
    assert capped[0] == ANNUAL_NIGHTS_CAP
    assert abs(ANNUAL_NIGHTS_CAP - 255.5) < 1e-9

    missing = booked_nights_ltm(np.array([0.0, np.nan]), np.array([2.0, 2.0]), 3.0)
    assert np.isnan(missing).all()


def test_grade_cuts_and_gross_yield():
    cuts = {"p20": 1000.0, "p40": 2000.0, "p60": 3000.0, "p80": 4000.0}
    assert grade_revenue(4000, cuts)["letter"] == "A"
    assert grade_revenue(3999, cuts)["words"] == "Above average"
    assert grade_revenue(3000, cuts)["letter"] == "B"
    assert grade_revenue(2000, cuts)["letter"] == "C"
    assert grade_revenue(1000, cuts)["letter"] == "D"
    assert grade_revenue(999, cuts)["letter"] == "F"
    assert annual_revenue(100, 80) == 8000
    assert gross_yield(8000, 200000) == 0.04
    assert gross_yield(8000, None) is None
    assert gross_yield(8000, 0) is None


def test_occupancy_features_exclude_the_label_inputs():
    assert OCCUPANCY_SEED not in OLD_SEEDS
    for _name, columns in FEATURE_LADDER:
        assert set(columns).isdisjoint(FORBIDDEN_COLUMNS)
        assert_columns_allowed(columns)
        assert "number_of_reviews_ltm" not in columns
        assert "minimum_nights_capped" not in columns
        assert "price_clean" not in columns


def test_occupancy_holdout_matches_seed_31_and_blocks_neighbours():
    saved = json.loads(PATH.read_text())
    assert saved["occupancy_seed"] == OCCUPANCY_SEED == 31
    assert "price" not in json.dumps(saved["cities"])
    from data.ingest import load_listings

    recomputed = computed_ids(tuple(saved["cities"]))
    for city, frozen in saved["cities"].items():
        assert frozen == recomputed[city]
        frame, _stats = load_listings(city)
        observed = population(frame)
        split = occupancy_split(
            observed["latitude"].to_numpy(dtype=float),
            observed["longitude"].to_numpy(dtype=float),
            observed["host_id"].to_numpy(),
        )
        assert sorted(listing_ids(observed.iloc[split["test_idx"]]["id"]), key=int) == frozen
        audit = split_audit(observed, split["development_idx"], split["test_idx"])
        assert audit["passed"], audit


def test_occupancy_report_matches_the_predictions_and_the_bar():
    payload = json.loads(METRICS.read_text())
    preds = pd.read_csv(PREDICTIONS)
    assert payload["smoke"] is False
    assert payload["seed"] == 31
    for city, report in payload["cities"].items():
        part = preds[preds["city"] == city]
        hold = report["holdout"]
        recomputed = regression_metrics(part["y_nights"].to_numpy(), part["pred_nights"].to_numpy())
        for key in ("r2", "mae", "rmse", "n"):
            assert abs(recomputed[key] - hold["model"][key]) < 1e-6
        area = regression_metrics(part["y_nights"].to_numpy(), part["baseline_area"].to_numpy())
        assert abs(area["mae"] - hold["baseline_area"]["mae"]) < 1e-6
        revenue = regression_metrics(part["y_revenue"].to_numpy(), part["pred_revenue"].to_numpy())
        assert abs(revenue["mae"] - hold["revenue"]["mae"]) < 1e-6
        assert abs(revenue["r2"] - hold["revenue"]["r2"]) < 1e-6
        np.testing.assert_allclose(
            part["pred_revenue"].to_numpy(),
            part["pred_price"].to_numpy() * part["pred_nights"].to_numpy(),
            rtol=1e-6,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            part["y_revenue"].to_numpy(),
            part["price"].to_numpy() * part["y_nights"].to_numpy(),
            rtol=1e-6,
            atol=1e-4,
        )
        expected = cleared_regression(
            hold["model"]["mae"], hold["baseline_area"]["mae"], hold["model"]["r2"]
        )
        assert hold["cleared_bar"] is expected
        klass = hold["classification"]
        assert klass["cleared_bar"] is cleared_classification(
            klass["accuracy"],
            klass["baseline_accuracy"],
            klass["balanced_accuracy"],
            klass["baseline_balanced_accuracy"],
        )
        assert set(payload["protocol"]["columns"]).isdisjoint(FORBIDDEN_COLUMNS)
