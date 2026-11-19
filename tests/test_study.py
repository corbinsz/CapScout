"""Rules and parsers for the simple spatial study. No full-city fit."""

import numpy as np
import pandas as pd
import pytest
from data.clean import apply_filters, parse_bathrooms, parse_price_series
from evaluation.study_models import (
    AreaMedian,
    dollar_effects,
    make_catboost,
    make_xgb,
    shap_means,
)
from evaluation.study_protocol import (
    CONFIRM_SEEDS,
    FORBIDDEN_SEEDS,
    PRIMARY_SEED,
    choose_nights_target,
    choose_params,
    choose_shipped,
    choose_simplest,
    nights_cleared,
    nights_selection_reason,
    selection_reason,
)
from evaluation.study_split import audit_split, cell_ids, choose_test_cells


def test_seeds_are_new():
    assert PRIMARY_SEED not in FORBIDDEN_SEEDS
    assert set(CONFIRM_SEEDS).isdisjoint(FORBIDDEN_SEEDS)
    assert PRIMARY_SEED not in CONFIRM_SEEDS


def test_price_parser_keeps_dollars_and_rejects_two_dots():
    parsed = parse_price_series(pd.Series(["$1,234.00", "97.00", "1.2.3", "", None]))
    assert parsed.tolist()[:2] == [1234.0, 97.0]
    assert np.isnan(parsed.iloc[2])
    assert np.isnan(parsed.iloc[3])
    assert np.isnan(parsed.iloc[4])


def test_half_bath_is_half():
    baths = parse_bathrooms(
        pd.Series([np.nan, 2.0, np.nan]),
        pd.Series(["Private half-bath", "2 baths", "1 shared bath"]),
    )
    assert baths.tolist() == [0.5, 2.0, 1.0]


def test_duplicate_id_keeps_the_reviewed_copy():
    raw = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "host_id": [9, 9, 8],
            "neighbourhood_cleansed": ["A", "A", "A"],
            "latitude": [30.27, 30.27, 30.28],
            "longitude": [-97.74, -97.74, -97.75],
            "property_type": ["House", "House", "House"],
            "room_type": ["Entire home/apt", "Entire home/apt", "Entire home/apt"],
            "accommodates": [2, 2, 2],
            "bathrooms": [1, 1, 1],
            "bathrooms_text": ["1 bath", "1 bath", "1 bath"],
            "bedrooms": [1, 1, 1],
            "beds": [1, 1, 1],
            "price": ["$100.00", "$100.00", "$120.00"],
            "minimum_nights": [2, 2, 2],
            "number_of_reviews": [1, 10, 4],
            "number_of_reviews_ltm": [1, 8, 3],
            "review_scores_rating": [5, 5, 5],
            "host_is_superhost": ["t", "t", "f"],
            "amenities": ["[]", "[]", "[]"],
            "availability_365": [100, 100, 100],
        }
    )
    frame, report = apply_filters(raw, "austin")
    assert set(frame["id"].tolist()) == {1, 2}
    kept = frame.loc[frame["id"] == 1, "number_of_reviews_ltm"].iloc[0]
    assert kept == 8
    removed = report["steps"][1]
    assert removed["id"] == "duplicate_id"
    assert removed["removed"] == 1


def test_choose_simplest_prefers_the_simple_model_inside_two_percent():
    # Best is xgboost at 100. Linear at 101 is inside 2%. Area median at 130 is not.
    chosen = choose_simplest(
        [
            ("area_median", 130.0),
            ("linear", 101.0),
            ("ridge", 100.5),
            ("random_forest", 100.2),
            ("xgboost", 100.0),
        ]
    )
    assert chosen == "linear"
    assert "Linear regression" in selection_reason(
        [
            ("area_median", 130.0),
            ("linear", 101.0),
            ("ridge", 100.5),
            ("random_forest", 100.2),
            ("xgboost", 100.0),
        ],
        chosen,
    )


def test_choose_shipped_needs_a_five_percent_win():
    # 96 is only 4% under 100, and 96 is not 5% under the blend either.
    assert choose_shipped("xgboost", 100.0, 96.0, 96.0) == "xgboost"
    # CatBoost clears 5%. The blend does not clear another 5% under CatBoost.
    assert choose_shipped("xgboost", 100.0, 94.0, 93.0) == "catboost"
    # The blend clears 5% under whichever single model is ahead.
    assert choose_shipped("xgboost", 100.0, 94.0, 89.0) == "blend"
    assert choose_shipped("xgboost", 100.0, 96.0, 95.0) == "blend"


def test_nights_bar_and_target_and_params_are_fixed_rules():
    assert nights_cleared(90, 100, 0.10) is True
    assert nights_cleared(91, 100, 0.20) is False
    assert nights_cleared(80, 100, 0.09) is False
    assert choose_nights_target(100, 98.1) == "raw"
    assert choose_nights_target(100, 98.0) == "log"
    chosen = choose_params(
        [
            {"mae": 10.0, "simplicity": [16, -5], "params": {"max_depth": 16}},
            {"mae": 10.05, "simplicity": [8, -20], "params": {"max_depth": 8}},
        ]
    )
    assert chosen["params"]["max_depth"] == 8
    text = nights_selection_reason([("linear", 70.0), ("xgboost", 60.0)], "linear")
    assert "nights" in text
    assert "$" not in text


def test_cells_do_not_depend_on_row_order_and_the_split_is_separated():
    lat = np.array([30.27, 30.27, 30.40, 30.40])
    lon = np.array([-97.74, -97.74, -97.90, -97.90])
    one = cell_ids(lat, lon, "austin")
    two = cell_ids(lat[::-1], lon[::-1], "austin")
    assert list(one) == list(two[::-1])
    cells = np.array([10, 10, 20, 20, 30, 30, 40, 40, 50, 50])
    first = choose_test_cells(cells, 0.2, 17)
    shuffled = choose_test_cells(cells[::-1], 0.2, 17)
    assert first == shuffled
    frame = pd.DataFrame(
        {
            "cell": [1, 1, 2, 2, 3, 3],
            "host_id": ["a", "b", "c", "d", "e", "f"],
            "latitude": [30.27, 30.271, 30.40, 30.401, 30.55, 30.551],
            "longitude": [-97.74, -97.741, -97.90, -97.901, -98.10, -98.101],
        }
    )
    audit = audit_split(frame, np.array([0, 1]), np.array([4, 5]))
    assert audit["passed"]
    close = audit_split(frame, np.array([0, 1]), np.array([1, 2]))
    assert close["passed"] is False


def test_tiny_tree_fit_explains_in_dollars_and_catboost_fits():
    rng = np.random.default_rng(17)
    n = 80
    bedrooms = rng.integers(1, 4, size=n)
    frame = pd.DataFrame(
        {
            "bedrooms": bedrooms,
            "bathrooms": rng.choice([1.0, 2.0], size=n),
            "latitude": 30.27 + rng.normal(0, 0.01, size=n),
        }
    )
    y = 80 + 40 * bedrooms + rng.normal(0, 5, size=n)
    columns = ["bedrooms", "bathrooms", "latitude"]
    model = make_xgb(columns, 2, 1, kind="log", high=None)
    model.fit(frame.iloc[:60], y[:60])
    pred = model.predict(frame.iloc[60:])
    assert np.isfinite(pred).all()
    assert pred.min() > 0
    effects = dollar_effects(model, frame, np.arange(60, 80), limit=20)
    assert effects[0]["feature"] == "bedrooms"
    shap = shap_means(model, frame, np.arange(60, 80), limit=20)
    assert shap[0]["feature"] == "bedrooms"
    cat = make_catboost(columns, 2, kind="log", high=None)
    cat.fit(frame.iloc[:60], y[:60])
    cat_pred = cat.predict(frame.iloc[60:])
    assert np.isfinite(cat_pred).all()
    area = AreaMedian().fit(
        pd.Series(["A"] * 60),
        pd.Series(["Entire home/apt"] * 60),
        y[:60],
    )
    guess = area.predict(pd.Series(["A"] * 5), pd.Series(["Entire home/apt"] * 5))
    assert np.isfinite(guess).all()


def test_address_lookup_rejects_empty_misses_and_far_pins():
    from app.locate import LocateError, check_pin, locate_address

    with pytest.raises(LocateError, match="Type an address"):
        locate_address("  ", "austin", fetch=lambda *_args: {})
    with pytest.raises(LocateError, match="did not respond"):

        def _boom(*_args):
            raise TimeoutError("down")

        locate_address("1100 Congress Ave", "austin", fetch=_boom)
    with pytest.raises(LocateError, match="No match"):
        locate_address("nowhere", "austin", fetch=lambda *_args: {"result": {"addressMatches": []}})
    payload = {
        "result": {
            "addressMatches": [
                {"matchedAddress": "Far away", "coordinates": {"y": 31.5, "x": -97.7}}
            ]
        }
    }
    with pytest.raises(LocateError, match="60 km"):
        locate_address("far", "austin", fetch=lambda *_args: payload)
    check_pin(30.2672, -97.7431, "austin")
    with pytest.raises(LocateError):
        check_pin(31.5, -97.7, "austin")


def test_nearby_comps_prefer_the_same_room_and_are_not_required_to_score():
    from models.serve import nearby_comps, score_home

    frame = pd.DataFrame(
        {
            "latitude": [30.27, 30.28, 30.50],
            "longitude": [-97.74, -97.75, -97.90],
            "price_clean": [100.0, 200.0, 300.0],
            "bedrooms": [2, 2, 4],
            "bathrooms": [1, 1, 2],
            "room_type": ["Entire home/apt", "Private room", "Entire home/apt"],
            "neighbourhood_cleansed": ["A", "B", "C"],
        }
    )
    comps = nearby_comps(frame, 30.2672, -97.7431, "Entire home/apt", 2, n=1)
    assert comps.iloc[0]["price_clean"] == 100.0
    scored = score_home(
        "austin",
        {
            "latitude": 30.2672,
            "longitude": -97.7431,
            "bedrooms": 2,
            "bathrooms": 1.0,
            "accommodates": 4,
            "room_type": "Entire home/apt",
            "property_type": "House",
        },
    )
    assert scored["nightly_rate"] > 0
    assert scored["interval_low"] >= 0
    assert scored["interval_high"] >= scored["interval_low"]
    assert scored["booked_nights"] >= 0
    assert scored["annual_revenue"] == pytest.approx(
        scored["nightly_rate"] * scored["booked_nights"]
    )
    assert scored["grade"] in {"A", "B", "C", "D", "F"}
    assert scored["reasons"]
    assert scored["nights_rough_estimate"] is True
    assert all("$" in item["sentence"] for item in scored["reasons"])
