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
    choose_blend,
    choose_depth,
    choose_library,
    choose_monotonic,
    choose_quality,
    choose_simpler,
    feature_label,
)

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "reports" / "metrics.json"
CSV_PATH = ROOT / "reports" / "locked_test_predictions.csv"
FINDINGS = ROOT / "reports" / "FINDINGS.md"
README = ROOT / "README.md"


def _load():
    return json.loads(JSON_PATH.read_text()), pd.read_csv(CSV_PATH)


def test_predictions_match_the_json():
    payload, preds = _load()
    assert payload["smoke"] is False
    assert payload["audit_passed"] is True
    for city, report in payload["cities"].items():
        part = preds[preds["city"] == city]
        recomputed = regression_metrics(part["y_true"].to_numpy(), part["pred_gbm"].to_numpy())
        saved = report["locked_test"]["gbm"]
        for key in ("r2", "mae", "rmse", "mape", "median_ape", "n"):
            assert abs(recomputed[key] - saved[key]) < 1e-6, (city, key, recomputed, saved)
        interval = interval_metrics(part["y_true"], part["pred_lo"], part["pred_hi"])
        saved_interval = report["locked_test"]["interval"]
        assert abs(interval["coverage"] - saved_interval["coverage"]) < 1e-6
        assert report["audit"]["split"]["passed"] is True


def test_prose_quotes_the_locked_test():
    payload, _preds = _load()
    findings = FINDINGS.read_text()
    readme = README.read_text()
    for city, report in payload["cities"].items():
        gbm = report["locked_test"]["gbm"]
        for text in (findings, readme):
            assert f"{gbm['r2']:.3f}" in text
            assert f"{gbm['mae']:.2f}" in text
            assert f"{gbm['rmse']:.2f}" in text
            assert f"{gbm['mape']:.1f}%" in text
            assert f"{gbm['median_ape']:.1f}%" in text
        dev = payload["development_comparison"][city]["gbm"]
        assert f"{dev['r2_mean']:.3f} ± {dev['r2_std']:.3f}" in findings
        assert f"{dev['mae_mean']:.2f} ± {dev['mae_std']:.2f}" in findings


def test_recorded_choices_follow_the_pre_set_rule():
    """The JSON winner is what the rule says, not a later edit."""
    payload, _preds = _load()
    protocol = payload["protocol"]
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


def test_leakage_demo_is_quoted_as_a_demo_not_a_result():
    payload, _preds = _load()
    demo = payload["leakage_demo"]
    findings = FINDINGS.read_text()
    assert f"{demo['cv_r2_against_smoothed_target']:.3f}" in findings
    assert f"{demo['cv_r2_against_actual_price']:.3f}" in findings
    assert "demonstration" in findings.lower()
