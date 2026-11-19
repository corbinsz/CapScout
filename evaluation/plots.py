"""Figures for the study. One chart, one question."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from features.columns import pretty

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_ablation(ablation: dict, path: Path) -> None:
    """Mean development MAE by feature prefix and city."""
    cities = list(ablation)
    sets = list(next(iter(ablation.values())))
    x = np.arange(len(sets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, city in enumerate(cities):
        means = [ablation[city][name]["mae_mean"] for name in sets]
        stds = [ablation[city][name]["mae_std"] for name in sets]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=city, capsize=2)
    ax.set_xticks(
        x, [name.replace("plus_", "+ ").replace("_", " ") for name in sets], rotation=15, ha="right"
    )
    ax.set_ylabel("Development MAE ($/night)")
    ax.set_title("Which feature group actually reduces error")
    ax.legend()
    _save(fig, path)


def plot_model_comparison(comparison: dict, path: Path, models: list[str] | None = None) -> None:
    cities = list(comparison)
    default = ["baseline_room", "baseline_room_bed", "ridge", "gbm"]
    models = models or [name for name in default if name in next(iter(comparison.values()))]
    labels = {
        "baseline_room": "Neighbourhood × room",
        "baseline_room_bed": "× bedroom bin",
        "ridge": "Ridge",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "v2": "v2 specification",
        "v3": "v3 specification",
        "gbm": "Selected",
    }
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, city in enumerate(cities):
        means = [comparison[city][m]["mae_mean"] for m in models]
        stds = [comparison[city][m]["mae_std"] for m in models]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=city, capsize=2)
    ax.set_xticks(x, [labels.get(name, name) for name in models], rotation=15, ha="right")
    ax.set_ylabel("Development MAE ($/night)")
    ax.set_title("Baselines, ridge, and the selected GBM")
    ax.legend()
    _save(fig, path)


def plot_learning_curves(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for city, rows in curves.items():
        frac = [r["fraction"] for r in rows]
        ax.plot(frac, [r["valid_mae"] for r in rows], marker="o", label=f"{city} validation")
        ax.plot(
            frac, [r["train_mae"] for r in rows], marker="o", linestyle="--", label=f"{city} train"
        )
    ax.set_xlabel("Fraction of development training cells")
    ax.set_ylabel("MAE ($/night)")
    ax.set_title("Learning curve on a development split (not the test)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def plot_pred_vs_actual(preds: pd.DataFrame, path: Path) -> None:
    cities = list(preds["city"].unique())
    fig, axes = plt.subplots(1, len(cities), figsize=(4.2 * len(cities), 4.2), sharey=False)
    if len(cities) == 1:
        axes = [axes]
    for ax, city in zip(axes, cities, strict=True):
        part = preds[preds["city"] == city]
        ax.scatter(part["y_true"], part["pred_gbm"], s=8, alpha=0.25, linewidths=0)
        limit = float(np.quantile(part["y_true"], 0.99))
        ax.plot([0, limit], [0, limit], color="black", linewidth=1)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("Actual nightly price ($)")
        ax.set_ylabel("Predicted ($)")
        ax.set_title(city)
    fig.suptitle("Final holdout: predicted vs actual", y=1.02)
    _save(fig, path)


def plot_residuals(preds: pd.DataFrame, path: Path) -> None:
    cities = list(preds["city"].unique())
    fig, axes = plt.subplots(1, len(cities), figsize=(4.2 * len(cities), 4.2))
    if len(cities) == 1:
        axes = [axes]
    for ax, city in zip(axes, cities, strict=True):
        part = preds[preds["city"] == city]
        resid = part["y_true"] - part["pred_gbm"]
        ax.scatter(part["pred_gbm"], resid, s=8, alpha=0.25, linewidths=0)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Predicted nightly price ($)")
        ax.set_ylabel("Actual − predicted ($)")
        ax.set_title(city)
    fig.suptitle("Final holdout residuals", y=1.02)
    _save(fig, path)


def plot_shap(shap_rows: dict, path: Path) -> None:
    cities = list(shap_rows)
    fig, axes = plt.subplots(1, len(cities), figsize=(4.4 * len(cities), 5.2))
    if len(cities) == 1:
        axes = [axes]
    for ax, city in zip(axes, cities, strict=True):
        pairs = shap_rows[city][:12][::-1]
        ax.barh([pretty(p["feature"]) for p in pairs], [p["mean_abs_shap"] for p in pairs])
        ax.set_xlabel("Mean |SHAP| on log price")
        ax.set_title(city)
    fig.suptitle("What the final-holdout model uses", y=1.02)
    _save(fig, path)


def plot_calibration(preds: pd.DataFrame, path: Path) -> None:
    """Does the 80% interval cover 80% in cheap listings and in expensive ones?"""
    cities = list(preds["city"].unique())
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for city in cities:
        part = preds[preds["city"] == city].copy()
        part["bin"] = pd.qcut(part["pred_gbm"], 8, duplicates="drop")
        rows = []
        for _label, group in part.groupby("bin", observed=True):
            covered = (group["y_true"] >= group["pred_lo"]) & (group["y_true"] <= group["pred_hi"])
            rows.append((float(group["pred_gbm"].median()), float(covered.mean())))
        ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label=city)
    ax.axhline(0.80, color="black", linewidth=1, linestyle="--", label="nominal 80%")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Median predicted nightly price in the bin ($)")
    ax.set_ylabel("Share of listings inside the interval")
    ax.set_title("Final holdout: interval coverage by predicted price")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_pdp(curves: dict, path: Path) -> None:
    """Partial dependence on the development rows. One feature, three cities."""
    if not curves:
        return
    features = list(next(iter(curves.values())))
    fig, axes = plt.subplots(1, len(features), figsize=(4.3 * len(features), 4.2), squeeze=False)
    for ax, feature in zip(axes[0], features, strict=True):
        for city, city_curves in curves.items():
            rows = city_curves.get(feature) or []
            if not rows:
                continue
            ax.plot(
                [r["value"] for r in rows], [r["mean_pred"] for r in rows], marker="o", label=city
            )
        ax.set_title(pretty(feature))
        ax.set_ylabel("Mean predicted nightly rate ($)")
        ax.set_xlabel("Feature value, other columns held")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Partial dependence on the development set", y=1.03)
    _save(fig, path)
