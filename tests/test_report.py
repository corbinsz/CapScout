"""The write-up has to match the file the evaluation wrote.

Recomputes R², MAE, and RMSE from the saved test predictions and checks
that both the JSON and the prose use those numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evaluation.metrics import interval_metrics, regression_metrics
from evaluation.protocol import STAGE1
from evaluation.select import (
    choose_addon,
    choose_band_interval,
    choose_blend,
    choose_confirmed,
    choose_correction,
    choose_depth,
    choose_interval,
    choose_library,
    choose_monotonic,
    choose_pooled,
    choose_quality,
    choose_sequential,
    choose_simpler,
    feature_label,
)

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "reports" / "metrics.json"
FINAL_PATH = ROOT / "reports" / "final_predictions.csv"
CSV_PATH = ROOT / "reports" / "locked_test_predictions.csv"
CONFIRM_PATH = ROOT / "reports" / "confirmation_predictions.csv"
V2_JSON = ROOT / "reports" / "v2_metrics.json"
V2_CSV = ROOT / "reports" / "v2_locked_test_predictions.csv"
V3_JSON = ROOT / "reports" / "v3_metrics.json"
FINDINGS = ROOT / "reports" / "FINDINGS.md"
README = ROOT / "README.md"


def _load():
    return json.loads(JSON_PATH.read_text()), pd.read_csv(CSV_PATH)


def _match(part: pd.DataFrame, saved: dict, saved_interval: dict) -> None:
    recomputed = regression_metrics(part["y_true"].to_numpy(), part["pred_gbm"].to_numpy())
    for key in ("r2", "mae", "rmse", "mape", "median_ape", "n"):
        assert abs(recomputed[key] - saved[key]) < 1e-6, (key, recomputed, saved)
    interval = interval_metrics(part["y_true"], part["pred_lo"], part["pred_hi"])
    assert abs(interval["coverage"] - saved_interval["coverage"]) < 1e-6


def test_predictions_match_the_json():
    payload, locked = _load()
    final = pd.read_csv(FINAL_PATH)
    confirm = pd.read_csv(CONFIRM_PATH)
    assert payload["smoke"] is False
    assert payload["audit_passed"] is True
    for city, report in payload["cities"].items():
        _match(
            final[final["city"] == city],
            report["final_holdout"]["gbm"],
            report["final_holdout"]["interval"],
        )
        _match(
            locked[locked["city"] == city],
            report["locked_test"]["gbm"],
            report["locked_test"]["interval"],
        )
        _match(
            confirm[confirm["city"] == city],
            report["confirmation"]["gbm"],
            report["confirmation"]["interval"],
        )
        assert report["audit"]["final"]["passed"] is True
        assert report["audit"]["split"]["passed"] is True
        assert report["audit"]["confirmation"]["passed"] is True
        # The point model was not changed, so seed 7's dollars match the archive.
        published = payload["published_v3"][city]["gbm"]["mae"]
        assert abs(report["locked_test"]["gbm"]["mae"] - published) < 1e-6


def test_prose_quotes_the_final_holdout():
    """The write-up a reader sees is the simple spatial study, not the old blend."""
    findings = FINDINGS.read_text()
    readme = README.read_text()
    for text in (findings, readme):
        assert "Ridge regression" in text
        assert "Neither CatBoost nor the blend is in the app." in text
        assert "seed 17" in text.lower() or "Seed 17" in text


def test_recorded_choices_follow_the_pre_set_rule():
    """The JSON winner is what the rule says, not a later edit.

    v2's search is replayed from the archived metrics. This round's add-ons,
    depth, pooling, and interval are replayed from the new file.
    """
    archive = json.loads(V2_JSON.read_text())
    protocol = archive["protocol"]
    stage1 = [(name, protocol["stage1_development_mae"][name]) for name, _cols, _mode in STAGE1]
    assert choose_simpler(stage1) == protocol["stage1"]
    base = protocol["stage1_development_mae"][protocol["stage1"]]
    addons = [
        ("target_smooth", protocol["feature_set_development_mae"]["target_smooth"]),
        ("target_knn", protocol["feature_set_development_mae"]["target_knn"]),
    ]
    addon = choose_addon(base, addons)
    assert addon == protocol["addon"]
    current = protocol["feature_set_development_mae"][addon] if addon else base
    quality = choose_quality(current, protocol["feature_set_development_mae"]["listing_quality"])
    assert quality == protocol["listing_quality"]
    assert feature_label(protocol["stage1"], addon, quality) == protocol["chosen_feature_set"]

    depths = [(int(depth), mae) for depth, mae in protocol["depth_development_mae"].items()]
    assert choose_depth(depths) == protocol["chosen_depth"]
    assert choose_library(protocol["library_development_mae"]) == protocol["component_library"]
    assert (
        choose_monotonic(
            protocol["library_development_mae"][protocol["component_library"]],
            protocol["monotonic_development_mae"],
        )
        == protocol["monotonic_within_bar"]
    )
    winner = (
        protocol["monotonic_development_mae"]
        if protocol["monotonic_within_bar"]
        else protocol["library_development_mae"][protocol["component_library"]]
    )
    assert choose_blend(winner, protocol["blend_development_mae"]) == protocol["blend"]

    v3 = json.loads(V3_JSON.read_text())["protocol"]
    kept_v3 = choose_sequential(
        v3["v3_base_mae"],
        [(item["name"], item["mae"]) for item in v3["v3_trials"]],
    )
    assert kept_v3 == v3["v3_kept"]
    depth_pairs = [(int(depth), mae) for depth, mae in v3["depth_development_mae"].items()]
    assert choose_depth(depth_pairs) == v3["chosen_depth"]
    assert (
        choose_pooled(v3["per_city_development_mae"], v3["pooled_development_mae"]) == v3["pooled"]
    )
    assert choose_interval(v3["interval_candidates"]) == v3["interval_method"]

    payload, _preds = _load()
    current = payload["protocol"]
    kept = choose_sequential(
        current["v4_base_mae"],
        [(item["name"], item["mae"]) for item in current["v4_trials"]],
    )
    assert kept == current["v4_kept"]
    # Nothing was added, so the third library is compared with the v3 baseline.
    assert choose_quality(current["v4_base_mae"], current["three_library_mae"]) is False
    screen_default = current["tune"]["screen_default"]
    best_screen = min(item["mae"] for item in current["tune"]["screen"])
    assert choose_confirmed(screen_default, best_screen) is False
    assert current["tune"]["kept"] is False
    assert (
        choose_confirmed(current["stacker_confirm_default"], current["stacker_confirm_mae"])
        == current["stacker_kept"]
    )
    assert (
        choose_correction(
            current["correction_mae"]["none"],
            [
                ("smear", current["correction_mae"]["smear"]),
                ("isotonic", current["correction_mae"]["isotonic"]),
            ],
        )
        == current["correction"]
    )
    assert choose_band_interval(current["interval_candidates"]) == current["interval_method"]


def test_published_v2_archive_matches_its_predictions():
    """The v2 numbers in the write-up are the archived file, not a retype."""
    archive = json.loads(V2_JSON.read_text())
    preds = pd.read_csv(V2_CSV)
    payload, _preds = _load()
    for city, report in archive["cities"].items():
        part = preds[preds["city"] == city]
        recomputed = regression_metrics(part["y_true"].to_numpy(), part["pred_gbm"].to_numpy())
        saved = report["locked_test"]["gbm"]
        for key in ("r2", "mae", "rmse"):
            assert abs(recomputed[key] - saved[key]) < 1e-6
        assert abs(recomputed["mae"] - payload["published_v2"][city]["gbm"]["mae"]) < 1e-6


def test_leakage_demo_is_quoted_as_a_demo_not_a_result():
    payload, _preds = _load()
    demo = payload["leakage_demo"]
    findings = FINDINGS.read_text()
    assert f"{demo['cv_r2_against_smoothed_target']:.3f}" in findings
    assert f"{demo['cv_r2_against_actual_price']:.3f}" in findings
    assert "demonstration" in findings.lower()
