"""The write-up has to match the file the evaluation wrote.

Recomputes R², MAE, and RMSE from the saved test predictions and checks
that both the JSON and the prose use those numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evaluation.metrics import regression_metrics
from evaluation.protocol import FEATURE_SETS, LIGHTGBM_MUST_BEAT, SIMPLER_WITHIN

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
    for city, report in payload["cities"].items():
        part = preds[preds["city"] == city]
        recomputed = regression_metrics(part["y_true"].to_numpy(), part["pred_gbm"].to_numpy())
        saved = report["locked_test"]["gbm"]
        for key in ("r2", "mae", "rmse", "n"):
            assert abs(recomputed[key] - saved[key]) < 1e-6, (city, key, recomputed, saved)


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
        dev = payload["development_comparison"][city]["gbm"]
        assert f"{dev['r2_mean']:.3f} ± {dev['r2_std']:.3f}" in findings
        assert f"{dev['mae_mean']:.2f} ± {dev['mae_std']:.2f}" in findings


def test_recorded_choices_follow_the_pre_set_rule():
    """The feature set and depth in the JSON are what the rule says, not a later edit."""
    payload, _preds = _load()
    protocol = payload["protocol"]
    means = [
        (name, protocol["feature_set_development_mae"][name])
        for name, _cols, _spatial in FEATURE_SETS
    ]
    best = min(mae for _, mae in means)
    chosen = next(name for name, mae in means if mae <= best * SIMPLER_WITHIN)
    assert chosen == protocol["chosen_feature_set"]

    depth_means = [(int(depth), mae) for depth, mae in protocol["depth_development_mae"].items()]
    depth_means.sort()
    best_depth = min(mae for _, mae in depth_means)
    chosen_depth = next(depth for depth, mae in depth_means if mae <= best_depth * SIMPLER_WITHIN)
    assert chosen_depth == protocol["chosen_depth"]

    assert protocol["library"] == "xgboost"
    assert (
        protocol["lightgbm_development_mae"]
        >= protocol["xgboost_development_mae"] * LIGHTGBM_MUST_BEAT
    )


def test_leakage_demo_is_quoted_as_a_demo_not_a_result():
    payload, _preds = _load()
    demo = payload["leakage_demo"]
    findings = FINDINGS.read_text()
    assert f"{demo['cv_r2_against_smoothed_target']:.3f}" in findings
    assert f"{demo['cv_r2_against_actual_price']:.3f}" in findings
    assert "demonstration" in findings.lower()
