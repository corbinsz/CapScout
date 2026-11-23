"""Drivers stay consistent with the fields a person actually set."""

import inspect

import numpy as np
import pandas as pd
import pytest
from evaluation.recheck import shipped_columns
from evaluation.run_study import run
from evaluation.study_protocol import choose_nights_target, improves_by
from models.artifacts import load_artifact
from models.explain import rate_reasons
from models.serve import score_home


class _Dollars:
    """A stand-in with a known formula, so the counterfactual can be checked."""

    def __init__(self):
        self.columns = [
            "bedrooms",
            "accommodates",
            "bedrooms_x_accommodates",
            "latitude",
            "distance_to_center_km",
        ]
        self.medians_ = {
            "bedrooms": 2.0,
            "accommodates": 4.0,
            "bedrooms_x_accommodates": 8.0,
            "latitude": 30.3,
            "distance_to_center_km": 8.0,
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        row = frame.iloc[0]

        def value(name: str) -> float:
            raw = row[name]
            if not np.isfinite(raw):
                return self.medians_[name]
            return float(raw)

        total = (
            100.0
            + 10.0 * value("bedrooms")
            + 5.0 * value("accommodates")
            + 2.0 * value("bedrooms_x_accommodates")
            + value("latitude")
            - 3.0 * value("distance_to_center_km")
        )
        return np.array([total])


def test_bedrooms_recomputes_the_interaction_instead_of_leaving_it_stale():
    model = _Dollars()
    row = pd.DataFrame(
        [
            {
                "bedrooms": 4.0,
                "accommodates": 8.0,
                "bedrooms_x_accommodates": 32.0,
                "latitude": 30.27,
                "distance_to_center_km": 1.0,
            }
        ]
    )
    reasons = {item["feature"]: item for item in rate_reasons(model, row, limit=10)}
    # 4 bedrooms and 8 guests, product 32. Median bedrooms is 2, so the
    # consistent product is 16, not the stale 32 and not the median product 8.
    # Dollars: 10 * (4-2) + 2 * (32-16) = 52.
    assert reasons["bedrooms"]["dollars"] == pytest.approx(52.0)
    assert "bedrooms_x_accommodates" not in reasons
    assert reasons["accommodates"]["dollars"] == pytest.approx(52.0)
    assert set(reasons) == {"bedrooms", "accommodates", "location"}
    assert "latitude" not in reasons
    assert "$52" in reasons["bedrooms"]["sentence"]


def test_a_typical_fact_is_not_listed():
    model = _Dollars()
    row = pd.DataFrame(
        [
            {
                "bedrooms": 2.0,
                "accommodates": 4.0,
                "bedrooms_x_accommodates": 8.0,
                "latitude": 30.3,
                "distance_to_center_km": 8.0,
            }
        ]
    )
    assert rate_reasons(model, row) == []


def test_two_percent_bar_matches_the_nights_boundary():
    assert improves_by(100.0, 98.0) is True
    assert improves_by(100.0, 98.1) is False
    assert choose_nights_target(100, 98.1) == "raw"
    assert choose_nights_target(100, 98.0) == "log"


def test_saved_model_columns_are_the_shipped_list():
    for city in ("austin", "san-diego", "seattle"):
        artifact = load_artifact(city)
        assert artifact["columns"] == shipped_columns(city)
        if city == "austin":
            assert "distance_to_coast_km" not in artifact["columns"]
        else:
            assert "distance_to_coast_km" in artifact["columns"]


def test_comparison_command_does_not_replace_artifacts_by_default():
    assert inspect.signature(run).parameters["write_artifacts"].default is False


def test_app_reasons_group_location_and_keep_the_interaction_out():
    scored = score_home(
        "austin",
        {
            "latitude": 30.2672,
            "longitude": -97.7431,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "accommodates": 6,
            "room_type": "Entire home/apt",
            "property_type": "House",
        },
    )
    features = [item["feature"] for item in scored["reasons"]]
    assert "bedrooms_x_accommodates" not in features
    assert "latitude" not in features
    assert "distance_to_center_km" not in features
    assert "location" in features
    assert scored["reasons"][0]["dollars"] != 0
