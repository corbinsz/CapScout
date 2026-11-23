"""The write-up quotes the frozen study, and the CSV recomputes it."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evaluation.metrics import regression_metrics
from evaluation.study_format import (
    nights_comparison_table,
    nights_confirmation_table,
    nights_result_table,
    price_comparison_table,
    price_confirmation_table,
    price_result_table,
)
from evaluation.study_models import bootstrap_metrics
from evaluation.study_overall import findings_overall, readme_overall_table
from evaluation.study_protocol import (
    CONFIRM_SEEDS,
    ELIGIBLE_MODELS,
    NIGHTS_FORBIDDEN,
    PRICE_FORBIDDEN,
    choose_params,
    choose_simplest,
    nights_cleared,
)
from models.artifacts import load_artifact

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "reports" / "study_metrics.json").read_text())
README = (ROOT / "README.md").read_text()
FINDINGS = (ROOT / "reports" / "FINDINGS.md").read_text()
PREDICTIONS = pd.read_csv(ROOT / "reports" / "study_predictions.csv")
CONFIRMATION = pd.read_csv(ROOT / "reports" / "study_confirmation.csv")


def test_overall_figures_match_the_holdout_and_the_docs():
    section = findings_overall(PREDICTIONS)
    table = readme_overall_table(PREDICTIONS)
    assert section in FINDINGS
    assert table in README
    assert "CC BY 4.0" in README
    assert "2025-09-16" in README
    assert "2025-03-16" in README
    assert "CC BY 4.0" in FINDINGS
    assert "2025-09-16" in FINDINGS
    assert "2025-03-16" in FINDINGS
    assert "$55.18" in section and "$84.59" in section
    assert "$61.60" in section and "$95.34" in section


def test_docs_quote_the_rendered_tables_and_the_reasons():
    tables = [
        price_comparison_table(METRICS),
        price_result_table(METRICS),
        price_confirmation_table(METRICS),
        nights_comparison_table(METRICS),
        nights_result_table(METRICS),
        nights_confirmation_table(METRICS),
    ]
    for table in tables:
        assert table in README
        assert table in FINDINGS
    for key in (
        "selection_reason",
        "nights_selection_reason",
        "nights_reason",
        "nights_target_sentence",
    ):
        assert METRICS[key] in README
        assert METRICS[key] in FINDINGS
    assert "0.973" in FINDINGS
    assert "0.385" in FINDINGS
    assert "demonstration" in FINDINGS.lower()


def test_holdout_csv_matches_the_json_and_the_rules():
    assert METRICS["primary_seed"] == 17
    assert METRICS["confirm_seeds"] == [19, 29, 37]
    assert METRICS["audit_passed"] is True
    assert set(METRICS["price_features"]).isdisjoint(PRICE_FORBIDDEN)
    assert set(METRICS["nights_features"]).isdisjoint(NIGHTS_FORBIDDEN)
    assert "number_of_reviews_ltm" not in METRICS["nights_features"]
    assert "log_reviews" not in METRICS["nights_features"]
    price_means = []
    for name in ELIGIBLE_MODELS:
        maes = []
        for city in ("austin", "san-diego", "seattle"):
            part = PREDICTIONS[PREDICTIONS["city"] == city]
            saved = METRICS["cities"][city]["price"]["holdout"][name]
            recomputed = regression_metrics(part["y_price"], part[f"price_{name}"])
            for key in ("r2", "mae", "median_ape"):
                assert abs(recomputed[key] - saved[key]) < 1e-6
            maes.append(saved["mae"])
            nights_saved = METRICS["cities"][city]["nights"]["holdout"][name]
            nights = regression_metrics(part["y_nights"], part[f"nights_{name}"])
            assert abs(nights["mae"] - nights_saved["mae"]) < 1e-6
            assert abs(nights["r2"] - nights_saved["r2"]) < 1e-6
        price_means.append((name, sum(maes) / len(maes)))
    assert choose_simplest(price_means) == METRICS["simple_model"] == METRICS["shipped_model"]
    assert METRICS["shipped_model"] == "ridge"
    assert set(METRICS["price_mean_mae"]) == set(ELIGIBLE_MODELS)
    nights_means = [(name, METRICS["nights_mean_mae"][name]) for name in ELIGIBLE_MODELS]
    assert choose_simplest(nights_means) == METRICS["nights_shipped_model"] == "linear"
    cleared = []
    for city in ("austin", "san-diego", "seattle"):
        hold = METRICS["cities"][city]["nights"]["holdout"]
        shipped = METRICS["nights_shipped_model"]
        flag = nights_cleared(hold[shipped]["mae"], hold["area_median"]["mae"], hold[shipped]["r2"])
        assert flag is METRICS["cities"][city]["nights"]["cleared_bar"]
        cleared.append(flag)
        for target in ("price", "nights"):
            block = METRICS["cities"][city][target]
            for name in ("ridge", "random_forest", "xgboost"):
                chosen = choose_params(block["trials"][name])
                assert chosen["params"] == block["params"][name]
    assert any(cleared) is False
    assert METRICS["nights_rough_estimate"] is True
    for city in ("austin", "san-diego", "seattle"):
        part = PREDICTIONS[PREDICTIONS["city"] == city]
        boot = bootstrap_metrics(
            part["y_price"].to_numpy(),
            part["price_ridge"].to_numpy(),
            part["price_area_median"].to_numpy(),
            part["cell"].to_numpy(),
            n_boot=1000,
            seed=17,
        )
        saved = METRICS["cities"][city]["price"]["bootstrap"]
        assert abs(boot["mae"]["low"] - saved["mae"]["low"]) < 1e-6
        assert abs(boot["r2"]["high"] - saved["r2"]["high"]) < 1e-6
    for city in ("austin", "san-diego", "seattle"):
        stored = METRICS["cities"][city]["price"]["confirmation"]["r2"]
        for seed, expected in zip(CONFIRM_SEEDS, stored, strict=True):
            part = CONFIRMATION[(CONFIRMATION["city"] == city) & (CONFIRMATION["seed"] == seed)]
            recomputed = regression_metrics(part["y_price"], part["pred_price"])
            assert abs(recomputed["r2"] - expected) < 1e-6


def test_artifact_is_the_simple_model_and_docs_have_no_deployed_site():
    artifact = load_artifact("austin")
    assert artifact["kind"] == "simple-study"
    assert artifact["library"] == "linear"
    assert artifact["nights_library"] == "linear"
    assert artifact["nights_rough_estimate"] is True
    assert type(artifact["model"].estimator.named_steps["model"]).__name__ == "LinearRegression"
    assert "attraction_index" in artifact["columns"]
    assert not (ROOT / "vercel.json").exists()
    adopted = pd.read_csv(ROOT / "reports" / "study_adopted_predictions.csv")
    shipped = readme_overall_table(adopted)
    assert shipped in README
    assert shipped in FINDINGS
    for path in (ROOT / "README.md", ROOT / "reports" / "FINDINGS.md", ROOT / "DATA.md"):
        text = path.read_text().lower()
        assert "vercel.app" not in text
        assert "onrender.com" not in text
        assert "streamlit.app" not in text
    app = (ROOT / "app" / "streamlit_app.py").read_text()
    assert "fonts.googleapis" not in app
    for path in (
        ROOT / "README.md",
        ROOT / "reports" / "FINDINGS.md",
        ROOT / "app" / "streamlit_app.py",
    ):
        text = path.read_text().lower()
        assert "catboost" not in text
        assert "neural net" not in text
        assert "neural_net" not in text
