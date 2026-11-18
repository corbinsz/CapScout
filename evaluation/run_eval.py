"""Run the study and write reports/metrics.json plus the figures.

    python -m evaluation.run_eval
    python -m evaluation.run_eval --smoke

Nothing in this file reads the locked test until `score_locked_test`.
Feature sets and hyperparameters are chosen from development splits only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from data.cities import CITIES
from data.ingest import load_listings
from models.artifacts import save_artifact

from evaluation.blocking import development_splits, locked_split
from evaluation.fitting import (
    design_matrices,
    fit_boosted,
    fit_predict_ridge,
    median_baseline,
    params_for_depth,
    predict_dollars,
    refit_boosted,
)
from evaluation.leakage_demo import austin_leakage_demo
from evaluation.metrics import regression_metrics
from evaluation.plots import (
    plot_ablation,
    plot_learning_curves,
    plot_model_comparison,
    plot_pred_vs_actual,
    plot_residuals,
    plot_shap,
)
from evaluation.protocol import (
    FEATURE_SET_WHY,
    FEATURE_SETS,
    INNER_SEEDS,
    LIGHTGBM_MUST_BEAT,
    SIMPLER_WITHIN,
    STUDY_RATIONALE,
    TEST_SEED,
    study_cities,
)

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
FIGURES = REPORTS / "figures"


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(len(arr))}


def _seed_summary(rows: list[dict], **match) -> dict[str, dict[str, float]]:
    picked = [row for row in rows if all(row.get(key) == value for key, value in match.items())]
    summary = {}
    for metric in ("r2", "mae", "rmse"):
        summary[f"{metric}_mean"] = _mean_std([row[metric] for row in picked])["mean"]
        summary[f"{metric}_std"] = _mean_std([row[metric] for row in picked])["std"]
    summary["n_seeds"] = len(picked)
    return summary


def _choose_named(means: list[tuple[str, float]], *, prefer_earlier: bool) -> str:
    """Pick the best name. Earlier names win when they are within 1% MAE."""
    best = min(mae for _, mae in means)
    if not prefer_earlier:
        return min(means, key=lambda item: item[1])[0]
    for name, mae in means:
        if mae <= best * SIMPLER_WITHIN:
            return name
    return means[-1][0]


def _fit_gbm_metrics(
    frame, split, train_idx, valid_idx, columns, use_spatial, library, depth
) -> dict:
    X_train, y_log, X_valid, y_valid, _encoder = design_matrices(
        frame, train_idx, valid_idx, columns, use_spatial
    )
    groups = split["groups"][train_idx]
    model = fit_boosted(library, params_for_depth(depth), X_train, y_log, groups)
    pred = predict_dollars(model, X_valid, library)
    metrics = regression_metrics(y_valid, pred)
    metrics["n_trees"] = getattr(model, "_capscout_trees", None)
    return metrics


def _reference_metrics(frame, train_idx, valid_idx, columns, use_spatial) -> dict[str, dict]:
    y = frame["price_clean"].to_numpy(dtype=float)
    neigh = frame["neighbourhood_cleansed"].to_numpy()
    room = frame["room_type"].to_numpy()
    beds = frame["bedrooms"].to_numpy(dtype=float)
    out = {
        "baseline_room": regression_metrics(
            y[valid_idx],
            median_baseline(neigh, room, beds, y, train_idx, valid_idx, by_bedroom=False),
        ),
        "baseline_room_bed": regression_metrics(
            y[valid_idx],
            median_baseline(neigh, room, beds, y, train_idx, valid_idx, by_bedroom=True),
        ),
    }
    X_train, y_log, X_valid, y_valid, _encoder = design_matrices(
        frame, train_idx, valid_idx, columns, use_spatial
    )
    out["ridge"] = regression_metrics(y_valid, fit_predict_ridge(X_train, y_log, X_valid))
    return out


def _learning_curve(
    frame, split, train_idx, valid_idx, columns, use_spatial, library, depth
) -> list[dict]:
    groups = split["groups"]
    train_groups = np.unique(groups[train_idx])
    rng = np.random.default_rng(0)
    rows = []
    for fraction in (0.15, 0.30, 0.50, 0.75, 1.0):
        take = max(8, int(round(len(train_groups) * fraction)))
        take = min(take, len(train_groups))
        chosen = set(rng.choice(train_groups, size=take, replace=False))
        subset = train_idx[np.isin(groups[train_idx], list(chosen))]
        X_train, y_log, X_valid, y_valid, _encoder = design_matrices(
            frame, subset, valid_idx, columns, use_spatial
        )
        model = fit_boosted(library, params_for_depth(depth), X_train, y_log, groups[subset])
        valid_pred = predict_dollars(model, X_valid, library)
        train_pred = predict_dollars(model, X_train, library)
        rows.append(
            {
                "fraction": fraction,
                "n_train": int(len(subset)),
                "valid_mae": regression_metrics(y_valid, valid_pred)["mae"],
                "train_mae": regression_metrics(np.expm1(y_log), train_pred)["mae"],
            }
        )
    return rows


def _shap_rows(model, X: pd.DataFrame, library: str) -> list[dict]:
    sample = X.sample(n=min(600, len(X)), random_state=TEST_SEED)
    if library == "xgboost":
        import xgboost as xgb

        contribs = model.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)
    else:
        contribs = model.predict(sample, pred_contrib=True)
    means = np.abs(np.asarray(contribs)[:, :-1]).mean(axis=0)
    order = np.argsort(-means)
    return [
        {"feature": str(sample.columns[i]), "mean_abs_shap": float(means[i])} for i in order[:15]
    ]


def _residual_bins(y_true: np.ndarray, pred: np.ndarray) -> list[dict]:
    frame = pd.DataFrame({"y": y_true, "pred": pred, "resid": y_true - pred})
    try:
        frame["bin"] = pd.qcut(frame["pred"], 5, duplicates="drop")
    except ValueError:
        return []
    rows = []
    for bin_label, part in frame.groupby("bin", observed=True):
        rows.append(
            {
                "predicted_bin": str(bin_label),
                "n": int(len(part)),
                "mae": float(part["resid"].abs().mean()),
                "mean_residual": float(part["resid"].mean()),
            }
        )
    return rows


def _room_errors(frame, test_idx, pred) -> list[dict]:
    part = frame.iloc[test_idx][["room_type", "price_clean"]].copy()
    part["pred"] = pred
    rows = []
    for room, group in part.groupby("room_type"):
        metrics = regression_metrics(group["price_clean"].to_numpy(), group["pred"].to_numpy())
        rows.append({"room_type": str(room), **metrics})
    return rows


def _prepare(city_key: str):
    frame, stats = load_listings(city_key)
    split = locked_split(
        frame["latitude"].to_numpy(dtype=float),
        frame["longitude"].to_numpy(dtype=float),
        frame["host_id"].to_numpy(),
    )
    inners = list(
        development_splits(
            split["groups"],
            split["development_idx"],
            frame["latitude"].to_numpy(dtype=float),
            frame["longitude"].to_numpy(dtype=float),
            frame["host_id"].to_numpy(),
            INNER_SEEDS,
        )
    )
    return frame, stats, split, inners


def run(smoke: bool = False) -> dict:
    cities = ("austin",) if smoke else study_cities()
    seeds = (0,) if smoke else INNER_SEEDS
    sets = FEATURE_SETS if not smoke else FEATURE_SETS[:2] + FEATURE_SETS[-1:]
    depths = (6,) if smoke else (4, 6, 8)
    compare_lgb = not smoke
    do_curve = not smoke

    # The protocol's MAX_TREES is the real cap. Smoke overrides the fitter
    # by shrinking it through the environment of this process only.
    if smoke:
        import evaluation.fitting as fitting
        import evaluation.protocol as protocol

        protocol.MAX_TREES = 40
        fitting.MAX_TREES = 40

    prepared = {}
    ablation_rows: list[dict] = []
    print("Loading and splitting", flush=True)
    for city in cities:
        frame, stats, split, inners = _prepare(city)
        if smoke:
            inners = [item for item in inners if item[0] in seeds]
        prepared[city] = {"frame": frame, "stats": stats, "split": split, "inners": inners}
        print(
            f"  {city}: {stats['rows_final']} rows, test {split['info']['n_test']}, "
            f"dev {split['info']['train_after']} "
            f"(dropped {split['info']['removed_host']} host, {split['info']['removed_buffer']} buffer)",
            flush=True,
        )

    print("Ablation on development splits", flush=True)
    for city, pack in prepared.items():
        for name, columns, use_spatial in sets:
            for seed, train_idx, valid_idx, _info in pack["inners"]:
                metrics = _fit_gbm_metrics(
                    pack["frame"],
                    pack["split"],
                    train_idx,
                    valid_idx,
                    columns,
                    use_spatial,
                    "xgboost",
                    6,
                )
                ablation_rows.append(
                    {"city": city, "seed": seed, "feature_set": name, "model": "xgboost", **metrics}
                )
                print(f"  {city} {name} seed {seed}: MAE {metrics['mae']:.2f}", flush=True)

    set_means = []
    for name, _columns, _spatial in sets:
        maes = [row["mae"] for row in ablation_rows if row["feature_set"] == name]
        set_means.append((name, float(np.mean(maes))))
    chosen_set = _choose_named(set_means, prefer_earlier=True)
    chosen_columns, use_spatial = next(
        (cols, spatial) for name, cols, spatial in sets if name == chosen_set
    )
    print(f"Chosen feature set: {chosen_set}", flush=True)

    print("Depth and library on the chosen features", flush=True)
    depth_rows: list[dict] = []
    for city, pack in prepared.items():
        for depth in depths:
            for seed, train_idx, valid_idx, _info in pack["inners"]:
                metrics = _fit_gbm_metrics(
                    pack["frame"],
                    pack["split"],
                    train_idx,
                    valid_idx,
                    chosen_columns,
                    use_spatial,
                    "xgboost",
                    depth,
                )
                depth_rows.append(
                    {
                        "city": city,
                        "seed": seed,
                        "depth": depth,
                        "library": "xgboost",
                        "model": "gbm",
                        **metrics,
                    }
                )
                print(
                    f"  {city} xgb depth {depth} seed {seed}: MAE {metrics['mae']:.2f}", flush=True
                )

    depth_means = []
    for depth in depths:
        maes = [row["mae"] for row in depth_rows if row["depth"] == depth]
        depth_means.append((str(depth), float(np.mean(maes))))
    chosen_depth = int(_choose_named(depth_means, prefer_earlier=True))

    lgb_rows: list[dict] = []
    if compare_lgb:
        for city, pack in prepared.items():
            for seed, train_idx, valid_idx, _info in pack["inners"]:
                metrics = _fit_gbm_metrics(
                    pack["frame"],
                    pack["split"],
                    train_idx,
                    valid_idx,
                    chosen_columns,
                    use_spatial,
                    "lightgbm",
                    chosen_depth,
                )
                lgb_rows.append(
                    {"city": city, "seed": seed, "library": "lightgbm", "model": "gbm", **metrics}
                )
                print(f"  {city} lgb seed {seed}: MAE {metrics['mae']:.2f}", flush=True)
        lgb_mae = float(np.mean([row["mae"] for row in lgb_rows]))
        xgb_mae = float(np.mean([row["mae"] for row in depth_rows if row["depth"] == chosen_depth]))
        library = "lightgbm" if lgb_mae < xgb_mae * LIGHTGBM_MUST_BEAT else "xgboost"
    else:
        lgb_mae = None
        xgb_mae = float(np.mean([row["mae"] for row in depth_rows if row["depth"] == chosen_depth]))
        library = "xgboost"
    print(f"Chosen model: {library} depth {chosen_depth}", flush=True)

    reference_rows: list[dict] = []
    curve_rows: dict[str, list] = {}
    print("Baselines, ridge, learning curves", flush=True)
    for city, pack in prepared.items():
        for seed, train_idx, valid_idx, _info in pack["inners"]:
            refs = _reference_metrics(
                pack["frame"], train_idx, valid_idx, chosen_columns, use_spatial
            )
            for model_name, metrics in refs.items():
                reference_rows.append({"city": city, "seed": seed, "model": model_name, **metrics})
        if do_curve:
            seed, train_idx, valid_idx, _info = pack["inners"][0]
            curve_rows[city] = _learning_curve(
                pack["frame"],
                pack["split"],
                train_idx,
                valid_idx,
                chosen_columns,
                use_spatial,
                library,
                chosen_depth,
            )
            print(f"  {city} learning curve done", flush=True)

    print("Locked test, once", flush=True)
    predictions = []
    city_reports = {}
    shap_report = {}
    for city, pack in prepared.items():
        frame = pack["frame"]
        split = pack["split"]
        dev = split["development_idx"]
        test = split["test_idx"]
        X_dev, y_log, X_test, y_test, encoder = design_matrices(
            frame, dev, test, chosen_columns, use_spatial
        )
        groups = split["groups"][dev]
        stopped = fit_boosted(library, params_for_depth(chosen_depth), X_dev, y_log, groups)
        n_trees = getattr(stopped, "_capscout_trees", None) or 200
        model = refit_boosted(library, params_for_depth(chosen_depth), n_trees, X_dev, y_log)
        gbm_pred = predict_dollars(model, X_test, library)
        ridge_pred = fit_predict_ridge(X_dev, y_log, X_test)
        y = frame["price_clean"].to_numpy(dtype=float)
        base_pred = median_baseline(
            frame["neighbourhood_cleansed"].to_numpy(),
            frame["room_type"].to_numpy(),
            frame["bedrooms"].to_numpy(dtype=float),
            y,
            dev,
            test,
            by_bedroom=False,
        )
        base_bed_pred = median_baseline(
            frame["neighbourhood_cleansed"].to_numpy(),
            frame["room_type"].to_numpy(),
            frame["bedrooms"].to_numpy(dtype=float),
            y,
            dev,
            test,
            by_bedroom=True,
        )
        locked = {
            "gbm": regression_metrics(y_test, gbm_pred),
            "ridge": regression_metrics(y_test, ridge_pred),
            "baseline_room": regression_metrics(y_test, base_pred),
            "baseline_room_bed": regression_metrics(y_test, base_bed_pred),
        }
        print(
            f"  {city} test GBM R2 {locked['gbm']['r2']:.3f} "
            f"MAE {locked['gbm']['mae']:.2f} RMSE {locked['gbm']['rmse']:.2f}",
            flush=True,
        )
        for i, row_idx in enumerate(test):
            predictions.append(
                {
                    "city": city,
                    "listing_id": (
                        str(frame.at[row_idx, "id"]) if "id" in frame.columns else str(row_idx)
                    ),
                    "y_true": float(y_test[i]),
                    "pred_gbm": float(gbm_pred[i]),
                    "pred_ridge": float(ridge_pred[i]),
                    "pred_baseline": float(base_pred[i]),
                    "pred_baseline_bed": float(base_bed_pred[i]),
                    "room_type": str(frame.at[row_idx, "room_type"]),
                    "neighbourhood": str(frame.at[row_idx, "neighbourhood_cleansed"]),
                    "distance_to_center_km": float(frame.at[row_idx, "distance_to_center_km"]),
                    "bedrooms": float(frame.at[row_idx, "bedrooms"]),
                }
            )
        dev_occ = frame.iloc[dev]
        dev_occ = dev_occ[np.isfinite(dev_occ["occupancy_rate"])]
        occ_map = (
            dev_occ.groupby("neighbourhood_cleansed")["occupancy_rate"]
            .median()
            .astype(float)
            .to_dict()
        )
        save_artifact(
            city,
            {
                "model": model,
                "library": library,
                "depth": chosen_depth,
                "n_trees": int(n_trees),
                "encoder": encoder,
                "columns": list(chosen_columns),
                "use_spatial": use_spatial,
                "feature_set": chosen_set,
                "occupancy_by_neighbourhood": {str(k): float(v) for k, v in occ_map.items()},
                "occupancy_global": (
                    float(dev_occ["occupancy_rate"].median()) if len(dev_occ) else 0.5
                ),
                "training_latitude": frame["latitude"].to_numpy(dtype=np.float32)[dev],
                "training_longitude": frame["longitude"].to_numpy(dtype=np.float32)[dev],
                "distance_p95": float(frame.iloc[dev]["distance_to_center_km"].quantile(0.95)),
            },
        )
        shap_report[city] = _shap_rows(model, X_test, library)
        city_reports[city] = {
            "display": CITIES[city]["display"],
            "snapshot": pack["stats"]["snapshot"],
            "cleaning": pack["stats"],
            "split": split["info"],
            "locked_test": locked,
            "residuals_by_predicted_quintile": _residual_bins(y_test, gbm_pred),
            "locked_test_by_room_type": _room_errors(frame, test, gbm_pred),
            "shap": shap_report[city],
        }

    ablation_summary = {}
    comparison = {}
    for city in cities:
        ablation_summary[city] = {}
        for name, _columns, _spatial in sets:
            maes = [
                row["mae"]
                for row in ablation_rows
                if row["city"] == city and row["feature_set"] == name
            ]
            r2s = [
                row["r2"]
                for row in ablation_rows
                if row["city"] == city and row["feature_set"] == name
            ]
            rmses = [
                row["rmse"]
                for row in ablation_rows
                if row["city"] == city and row["feature_set"] == name
            ]
            ablation_summary[city][name] = {
                "mae_mean": _mean_std(maes)["mean"],
                "mae_std": _mean_std(maes)["std"],
                "r2_mean": _mean_std(r2s)["mean"],
                "r2_std": _mean_std(r2s)["std"],
                "rmse_mean": _mean_std(rmses)["mean"],
                "rmse_std": _mean_std(rmses)["std"],
            }
        comparison[city] = {}
        for model_name in ("baseline_room", "baseline_room_bed", "ridge"):
            part = [
                row for row in reference_rows if row["city"] == city and row["model"] == model_name
            ]
            comparison[city][model_name] = {
                "mae_mean": _mean_std([r["mae"] for r in part])["mean"],
                "mae_std": _mean_std([r["mae"] for r in part])["std"],
                "r2_mean": _mean_std([r["r2"] for r in part])["mean"],
                "r2_std": _mean_std([r["r2"] for r in part])["std"],
                "rmse_mean": _mean_std([r["rmse"] for r in part])["mean"],
                "rmse_std": _mean_std([r["rmse"] for r in part])["std"],
            }
        gbm_part = [
            row
            for row in (lgb_rows if library == "lightgbm" else depth_rows)
            if row["city"] == city and (library == "lightgbm" or row["depth"] == chosen_depth)
        ]
        comparison[city]["gbm"] = {
            "mae_mean": _mean_std([r["mae"] for r in gbm_part])["mean"],
            "mae_std": _mean_std([r["mae"] for r in gbm_part])["std"],
            "r2_mean": _mean_std([r["r2"] for r in gbm_part])["mean"],
            "r2_std": _mean_std([r["r2"] for r in gbm_part])["std"],
            "rmse_mean": _mean_std([r["rmse"] for r in gbm_part])["mean"],
            "rmse_std": _mean_std([r["rmse"] for r in gbm_part])["std"],
        }

    leakage = austin_leakage_demo(prepared["austin"]["frame"]) if "austin" in prepared else None

    result = {
        "study_cities": list(cities),
        "rationale": STUDY_RATIONALE,
        "protocol": {
            "test_seed": TEST_SEED,
            "inner_seeds": list(seeds),
            "chosen_feature_set": chosen_set,
            "feature_set_why": FEATURE_SET_WHY,
            "feature_set_development_mae": {name: mae for name, mae in set_means},
            "chosen_depth": chosen_depth,
            "depth_development_mae": {name: mae for name, mae in depth_means},
            "xgboost_development_mae": xgb_mae,
            "lightgbm_development_mae": lgb_mae,
            "library": library,
            "use_spatial": use_spatial,
            "columns": list(chosen_columns)
            + (["neigh_log_price", "neigh_bed_log_price", "knn_log_price"] if use_spatial else []),
            "selection_rule": (
                "Prefixes are cumulative. The simplest prefix within 1% mean development "
                "MAE of the best prefix is kept. Shallower trees win the same way. "
                "LightGBM is used only if its development MAE is under 98% of XGBoost."
            ),
        },
        "cities": city_reports,
        "development_ablation": ablation_summary,
        "development_comparison": comparison,
        "learning_curves": curve_rows,
        "leakage_demo": leakage,
        "smoke": smoke,
    }
    return result, pd.DataFrame(predictions), shap_report


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate the nightly-price study.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Austin only, one seed, tiny trees. Not for the report.",
    )
    args = parser.parse_args()
    result, predictions, shap_report = run(smoke=args.smoke)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        out = REPORTS / "smoke_metrics.json"
        out.write_text(json.dumps(_jsonable(result), indent=2))
        print(f"Wrote {out}", flush=True)
        return

    (REPORTS / "metrics.json").write_text(json.dumps(_jsonable(result), indent=2))
    predictions.to_csv(REPORTS / "locked_test_predictions.csv", index=False)
    plot_ablation(result["development_ablation"], FIGURES / "ablation_mae.png")
    plot_model_comparison(result["development_comparison"], FIGURES / "model_comparison.png")
    if result["learning_curves"]:
        plot_learning_curves(result["learning_curves"], FIGURES / "learning_curves.png")
    plot_pred_vs_actual(predictions, FIGURES / "pred_vs_actual.png")
    plot_residuals(predictions, FIGURES / "residuals.png")
    plot_shap(shap_report, FIGURES / "shap.png")
    print("Wrote reports/metrics.json, locked_test_predictions.csv, and figures/", flush=True)


if __name__ == "__main__":
    main()
