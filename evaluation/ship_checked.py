"""Refit the model the adopted-column check selected, and write the app files.

The check is in the module docstring of ``evaluation.recheck`` plus one
holdout score. Linear regression had the lowest equal-city holdout error
on the adopted columns, and it is the simplest model inside 2% of that
error. This command fits that model on the frozen training rows, writes
the app artifacts, and replaces ``reports/study_adopted_predictions.csv``.

It does not refit the nights model. That file is copied from the artifact
that was already scored. Confirmation seeds are not re-run.

    python -m evaluation.ship_checked
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from data.cities import STUDY_CITIES
from models.artifacts import load_artifact, save_artifact
from models.grade import grade_revenue

from evaluation.metrics import interval_metrics, regression_metrics
from evaluation.recheck import shipped_columns, training_folds, with_extra_columns
from evaluation.study_data import prepare_city
from evaluation.study_models import (
    AreaMedian,
    apply_interval,
    bootstrap_metrics,
    fit_interval,
    make_forest,
    make_linear,
    make_ridge,
    make_xgb,
    oof_predict,
)
from evaluation.study_overall import bootstrap_overall, readme_overall_table
from evaluation.study_protocol import (
    BOOTSTRAP_SEED,
    INTERVAL_MIN_BIN,
    N_BOOTSTRAP,
    PRIMARY_SEED,
)
from evaluation.study_split import split_frame

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Hyperparameters chosen on the training cells during the recheck.
# Ridge keeps the alphas already published. Forest and XGBoost keep the
# simplest trial inside 1% of the best training-cell error.
RIDGE_ALPHA = {"austin": 100.0, "san-diego": 100.0, "seattle": 10.0}
FOREST = {"max_depth": 8, "min_samples_leaf": 5}
XGB = {"max_depth": 4, "min_child_weight": 20.0}


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    return value


def _builders(columns: list[str], city: str) -> dict:
    return {
        "linear": lambda: make_linear(columns, kind="log", high=None),
        "ridge": lambda: make_ridge(columns, RIDGE_ALPHA[city], kind="log", high=None),
        "random_forest": lambda: make_forest(
            columns, FOREST["max_depth"], FOREST["min_samples_leaf"], kind="log", high=None
        ),
        "xgboost": lambda: make_xgb(
            columns, XGB["max_depth"], XGB["min_child_weight"], kind="log", high=None
        ),
    }


def main() -> None:
    frames = {}
    splits = {}
    for city in STUDY_CITIES:
        frame, _report = prepare_city(city)
        frames[city] = with_extra_columns(frame)
        split = split_frame(frames[city], city, PRIMARY_SEED)
        split["folds"] = training_folds(frames[city], split["train_idx"])
        splits[city] = split

    rows = []
    city_payload = {}
    for city in STUDY_CITIES:
        frame = frames[city]
        columns = shipped_columns(city)
        y = frame["price_clean"].to_numpy(dtype=float)
        nights = frame["booked_nights"].to_numpy(dtype=float)
        train_idx = splits[city]["train_idx"]
        test_idx = splits[city]["test_idx"]
        previous = load_artifact(city)
        builders = _builders(columns, city)
        fitted = {
            name: builder().fit(frame.iloc[train_idx], y[train_idx])
            for name, builder in builders.items()
        }
        area = AreaMedian().fit(
            frame.iloc[train_idx]["neighbourhood_cleansed"],
            frame.iloc[train_idx]["room_type"],
            y[train_idx],
        )
        test = frame.iloc[test_idx]
        preds = {name: model.predict(test) for name, model in fitted.items()}
        preds["area_median"] = area.predict(test["neighbourhood_cleansed"], test["room_type"])
        oof = oof_predict(builders["linear"], frame, y, splits[city]["folds"])
        table = fit_interval(y[train_idx], oof[train_idx], INTERVAL_MIN_BIN)
        low, high = apply_interval(preds["linear"], table)
        coverage = interval_metrics(y[test_idx], low, high)
        boot = bootstrap_metrics(
            y[test_idx],
            preds["linear"],
            preds["area_median"],
            frame["cell"].to_numpy()[test_idx],
            n_boot=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )
        nights_model = previous["nights_model"]
        train_rate = fitted["linear"].predict(frame.iloc[train_idx])
        train_nights = nights_model.predict(frame.iloc[train_idx])
        train_revenue = train_rate * train_nights
        cuts = {
            f"p{pct}": float(np.quantile(train_revenue, pct / 100.0)) for pct in (20, 40, 60, 80)
        }
        test_nights = nights_model.predict(test)
        revenue_p = preds["linear"] * test_nights
        revenue_y = y[test_idx] * nights[test_idx]
        revenue = regression_metrics(revenue_y, revenue_p)
        letters_match = [
            grade_revenue(float(pred), cuts)["letter"]
            == grade_revenue(float(actual), cuts)["letter"]
            for pred, actual in zip(revenue_p, revenue_y, strict=True)
        ]
        artifact = dict(previous)
        artifact["library"] = "linear"
        artifact["model"] = fitted["linear"]
        artifact["columns"] = list(columns)
        artifact["medians"] = dict(fitted["linear"].medians_)
        artifact["interval"] = table
        artifact["interval_coverage"] = coverage["coverage"]
        artifact["interval_median_width"] = coverage["median_width"]
        artifact["price_holdout_r2"] = boot["r2"]["point"]
        artifact["price_holdout_mae"] = boot["mae"]["point"]
        artifact["price_baseline_mae"] = boot["baseline_mae"]["point"]
        artifact["grade_cuts"] = cuts
        save_artifact(city, artifact)

        for position, row_index in enumerate(test_idx):
            row = {
                "city": city,
                "listing_id": int(frame.iloc[row_index]["id"]),
                "cell": int(frame.iloc[row_index]["cell"]),
                "y_price": float(y[row_index]),
                "price_model": float(preds["linear"][position]),
                "price_area_median": float(preds["area_median"][position]),
                "price_linear": float(preds["linear"][position]),
                "price_ridge": float(preds["ridge"][position]),
                "price_random_forest": float(preds["random_forest"][position]),
                "price_xgboost": float(preds["xgboost"][position]),
            }
            rows.append(row)
        point = regression_metrics(y[test_idx], preds["linear"])
        base = regression_metrics(y[test_idx], preds["area_median"])
        city_payload[city] = {
            "n": int(len(test_idx)),
            "n_cells": int(frame.iloc[test_idx]["cell"].nunique()),
            "r2": point["r2"],
            "mae": point["mae"],
            "median_ape": point["median_ape"],
            "baseline_mae": base["mae"],
            "error_removed": (base["mae"] - point["mae"]) / base["mae"],
            "coverage": coverage["coverage"],
            "median_width": coverage["median_width"],
            "bootstrap": boot,
            "grade_cuts": cuts,
            "columns": list(columns),
            "revenue_r2": revenue["r2"],
            "revenue_mae": revenue["mae"],
            "grade_match": float(np.mean(letters_match)),
            "holdout_mae": {
                name: float(regression_metrics(y[test_idx], pred)["mae"])
                for name, pred in preds.items()
            },
        }
        print(
            f"{city} coverage {coverage['coverage']:.1%} width ${coverage['median_width']:.0f} "
            f"grade match {city_payload[city]['grade_match']:.1%} revenue R2 {revenue['r2']:.3f}",
            flush=True,
        )

    predictions = pd.DataFrame(rows)
    path = REPORTS / "study_adopted_predictions.csv"
    predictions.to_csv(path, index=False)
    overall = bootstrap_overall(predictions)
    payload = {
        "shipped_model": "linear",
        "selection": (
            "On the adopted columns, linear regression had the lowest equal-city "
            "holdout error. It is the simplest model inside 2% of that error."
        ),
        "cities": city_payload,
        "overall": overall,
    }
    (REPORTS / "study_adopted.json").write_text(json.dumps(_jsonify(payload), indent=2) + "\n")
    print(readme_overall_table(predictions))
    _slices(predictions)
    print("wrote", path)


def _slices(predictions: pd.DataFrame) -> None:
    y = predictions["y_price"].to_numpy()
    pred = predictions["price_model"].to_numpy()
    base = predictions["price_area_median"].to_numpy()
    for tol, label in ((0.20, "20"), (0.30, "30")):
        model_share = float(np.mean(np.abs(y - pred) / y <= tol))
        base_share = float(np.mean(np.abs(y - base) / y <= tol))
        print(f"within {label}% model {model_share:.4f} baseline {base_share:.4f}")
    edges = np.quantile(y, [0.25, 0.5, 0.75])
    bins = np.digitize(y, edges, right=True)
    for bin_id, name in enumerate(("lowest", "second", "third", "highest")):
        mask = bins == bin_id
        err = pred[mask] - y[mask]
        print(
            f"tier {name} n {int(mask.sum())} "
            f"rate {y[mask].min():.0f}-{y[mask].max():.0f} "
            f"mae {np.mean(np.abs(err)):.2f} bias {np.mean(err):.2f}"
        )
    top = y >= np.quantile(y, 0.9)
    err = pred[top] - y[top]
    share = float(np.sum(np.abs(pred - y)[top]) / np.sum(np.abs(pred - y)))
    print(
        f"top10 n {int(top.sum())} rate {y[top].min():.0f}-{y[top].max():.0f} "
        f"mae {np.mean(np.abs(err)):.2f} bias {np.mean(err):.2f} error share {share:.4f}"
    )


if __name__ == "__main__":
    main()
