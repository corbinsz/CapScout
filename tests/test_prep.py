"""Cleaning and the saved model, checked against the published files.

The snapshot tests skip when the Inside Airbnb files are not on disk.
CI downloads them before pytest. A checkout without that download still
runs the unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from data.cities import STUDY_CITIES
from data.ingest import raw_city_dir
from evaluation.study_data import collapse_building

ROOT = Path(__file__).resolve().parents[1]


def _snapshots_ready() -> bool:
    return all((raw_city_dir(city) / "listings.csv.gz").exists() for city in STUDY_CITIES)


@pytest.mark.skipif(not _snapshots_ready(), reason="Inside Airbnb snapshots are not downloaded")
def test_clean_counts_match_the_published_funnel():
    from data.clean import load_clean_city

    quality = json.loads((ROOT / "reports" / "data_quality.json").read_text())
    for city in STUDY_CITIES:
        _frame, report = load_clean_city(city)
        saved = quality["cities"][city]
        assert report["rows_raw"] == saved["rows_raw"]
        assert report["rows_final"] == saved["rows_final"]
        assert [step["id"] for step in report["steps"]] == [step["id"] for step in saved["steps"]]
        assert [step["removed"] for step in report["steps"]] == [
            step["removed"] for step in saved["steps"]
        ]


@pytest.mark.skipif(not _snapshots_ready(), reason="Inside Airbnb snapshots are not downloaded")
def test_frozen_ids_and_saved_model_match_the_published_holdout():
    from evaluation.study_data import prepare_city
    from evaluation.study_split import ensure_ids
    from models.artifacts import load_artifact

    frames = {}
    for city in STUDY_CITIES:
        frames[city], _report = prepare_city(city)
    ensure_ids(frames)

    published = pd.read_csv(ROOT / "reports" / "study_adopted_predictions.csv")
    for city in STUDY_CITIES:
        model = load_artifact(city)["model"]
        part = published[published["city"] == city]
        frame = frames[city].copy()
        frame["listing_id"] = frame["id"].astype("int64").astype(str)
        part = part.copy()
        part["listing_id"] = part["listing_id"].astype("int64").astype(str)
        merged = part.merge(frame, on="listing_id", how="inner", suffixes=("", "_row"))
        assert len(merged) == len(part)
        predicted = model.predict(merged)
        assert np.allclose(predicted, merged["price_model"].to_numpy(), rtol=0, atol=1e-6)
        assert np.allclose(merged["price_model"], merged["price_linear"], rtol=0, atol=1e-6)


def test_townhouse_cabin_and_guest_unit_count_as_a_house():
    frame = pd.DataFrame(
        {
            "is_house": [0, 1, 0, 0],
            "is_townhouse": [1, 0, 0, 0],
            "is_cabin": [0, 0, 1, 0],
            "is_cottage": [0, 0, 0, 0],
            "is_guest": [0, 0, 0, 1],
        }
    )
    collapsed = collapse_building(frame)
    assert collapsed["is_house"].tolist() == [1, 1, 1, 1]
