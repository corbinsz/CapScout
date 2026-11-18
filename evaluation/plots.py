"""Figures for the study. One chart, one question."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def plot_model_comparison(comparison: dict, path: Path) -> None:
    cities = list(comparison)
    models = ["baseline_room", "baseline_room_bed", "ridge", "gbm"]
    labels = ["Neighbourhood × room", "× bedroom bin", "Ridge", "GBM"]
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, city in enumerate(cities):
        means = [comparison[city][m]["mae_mean"] for m in models]
        stds = [comparison[city][m]["mae_std"] for m in models]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=city, capsize=2)
    ax.set_xticks(x, labels)
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
    fig.suptitle("Locked test: predicted vs actual", y=1.02)
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
    fig.suptitle("Locked test residuals", y=1.02)
    _save(fig, path)


def plot_shap(shap_rows: dict, path: Path) -> None:
    cities = list(shap_rows)
    fig, axes = plt.subplots(1, len(cities), figsize=(4.4 * len(cities), 5.2))
    if len(cities) == 1:
        axes = [axes]
    for ax, city in zip(axes, cities, strict=True):
        pairs = shap_rows[city][:12][::-1]
        ax.barh([p["feature"] for p in pairs], [p["mean_abs_shap"] for p in pairs])
        ax.set_xlabel("Mean |SHAP| on log price")
        ax.set_title(city)
    fig.suptitle("What the locked-test model uses", y=1.02)
    _save(fig, path)
