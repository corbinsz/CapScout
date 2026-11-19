"""Figures for the shipped study. Run from the repo root:

    python -m evaluation.study_figures

Predicted-versus-actual and the price-tier chart use the adopted ridge model.
The comparison chart is the holdout that chose ridge, before those features.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.artifacts import load_artifact

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
CITIES = ("austin", "san-diego", "seattle")
LABELS = {"austin": "Austin", "san-diego": "San Diego", "seattle": "Seattle"}


def _pred_vs_actual(adopted: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharex=False, sharey=False)
    for axis, city in zip(axes, CITIES, strict=True):
        part = adopted[adopted["city"] == city]
        y = part["y_price"].to_numpy()
        pred = part["price_ridge"].to_numpy()
        axis.scatter(y, pred, s=8, alpha=0.35, color="#0f6e56", linewidths=0)
        limit = float(np.quantile(np.concatenate([y, pred]), 0.98))
        axis.plot([0, limit], [0, limit], color="#1c1915", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_title(LABELS[city])
        axis.set_xlabel("Advertised rate")
        axis.set_ylabel("Predicted rate")
    figure.suptitle("Held-out homes: predicted nightly rate versus advertised rate", fontsize=12)
    figure.tight_layout()
    figure.savefig(FIGURES / "pred_vs_actual.png", dpi=140)
    plt.close(figure)


def _tiers(adopted: pd.DataFrame) -> None:
    y = adopted["y_price"].to_numpy()
    pred = adopted["price_ridge"].to_numpy()
    edges = np.quantile(y, [0.25, 0.5, 0.75])
    bins = np.digitize(y, edges, right=True)
    names = ["Lowest quartile", "Second", "Third", "Highest quartile", "Top 10%"]
    masks = [bins == bin_id for bin_id in range(4)]
    masks.append(y >= np.quantile(y, 0.9))
    mae, bias = [], []
    for mask in masks:
        err = pred[mask] - y[mask]
        mae.append(float(np.mean(np.abs(err))))
        bias.append(float(np.mean(err)))
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    positions = np.arange(len(names))
    axis.bar(positions, mae, color="#c4b49a", label="Average absolute error")
    axis.plot(positions, bias, color="#a33b24", marker="o", label="Bias (prediction − actual)")
    axis.axhline(0, color="#1c1915", linewidth=0.6)
    axis.set_xticks(positions, names)
    axis.set_ylabel("Dollars")
    axis.set_title("Error by advertised-price tier, all 3,648 held-out homes")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "error_by_tier.png", dpi=140)
    plt.close(figure)


def _coefficients() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.6))
    for axis, city in zip(axes, CITIES, strict=True):
        model = load_artifact(city)["model"]
        coef = np.asarray(model.estimator.named_steps["model"].coef_, dtype=float).ravel()
        percent = (np.exp(coef) - 1.0) * 100.0
        order = np.argsort(np.abs(percent))[-8:]
        axis.barh(
            [model.columns[i].replace("_", " ") for i in order],
            percent[order],
            color="#0f6e56",
        )
        axis.axvline(0, color="#1c1915", linewidth=0.6)
        axis.set_title(LABELS[city])
        axis.set_xlabel("% change in (1 + rate) per 1 SD")
    figure.suptitle("Ridge coefficients, standardized features", fontsize=12)
    figure.tight_layout()
    figure.savefig(FIGURES / "ridge_coefficients.png", dpi=140)
    plt.close(figure)


def _comparison(selection: pd.DataFrame) -> None:
    names = [
        ("price_area_median", "Local median rate"),
        ("price_ridge", "Ridge"),
        ("price_random_forest", "Random forest"),
        ("price_xgboost", "XGBoost"),
    ]
    means = []
    for column, _label in names:
        city_mae = []
        for city in CITIES:
            part = selection[selection["city"] == city]
            city_mae.append(float(np.mean(np.abs(part[column] - part["y_price"]))))
        means.append(float(np.mean(city_mae)))
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    colors = ["#c4b49a", "#0f6e56", "#6d6456", "#1c1915"]
    axis.bar([label for _column, label in names], means, color=colors)
    axis.set_ylabel("Average miss, dollars")
    axis.set_title("Each city once, on the columns used to choose the model")
    for position, value in enumerate(means):
        axis.text(position, value + 1.2, f"${value:.0f}", ha="center", fontsize=9)
    axis.set_ylim(0, max(means) * 1.18)
    figure.tight_layout()
    figure.savefig(FIGURES / "model_comparison.png", dpi=140)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    adopted = pd.read_csv(ROOT / "reports" / "study_adopted_predictions.csv")
    selection = pd.read_csv(ROOT / "reports" / "study_predictions.csv")
    _pred_vs_actual(adopted)
    _tiers(adopted)
    _coefficients()
    _comparison(selection)
    print("wrote", FIGURES)


if __name__ == "__main__":
    main()
