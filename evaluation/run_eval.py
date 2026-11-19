"""Run the study and write reports/metrics.json plus the figures.

    python -m evaluation.run_eval
    python -m evaluation.run_eval --smoke

v3 choices use development folds that exclude the confirmation holdout.
The locked test and the confirmation holdout are scored once, after that.
`--smoke` is a crash check, not a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from data.cities import CITIES
from data.external import ensure_external
from data.ingest import load_study_frame
from features.columns import (
    LOCATION_COLUMNS,
    SIZE_COLUMNS,
    pretty,
)
from models.artifacts import save_artifact
from models.explain import shap_matrix

from evaluation.audit import column_audit, split_audit
from evaluation.blocking import confirmation_split, development_splits, locked_split
from evaluation.confirm import ensure_confirm_ids
from evaluation.fitting import (
    design_matrices,
    fit_boosted,
    fit_predict_ridge,
    median_baseline,
    params_for_depth,
    predict_dollars,
    refit_boosted,
)
from evaluation.freeze import ensure_frozen_ids
from evaluation.intervals import conformal_radius, cqr_bounds
from evaluation.leakage_demo import austin_leakage_demo
from evaluation.metrics import interval_metrics, regression_metrics
from evaluation.plots import (
    plot_ablation,
    plot_calibration,
    plot_learning_curves,
    plot_model_comparison,
    plot_pdp,
    plot_pred_vs_actual,
    plot_residuals,
    plot_shap,
)
from evaluation.protocol import (
    CONFIRM_SEED,
    FEATURE_SET_WHY,
    INTERVAL_LEVEL,
    STUDY_RATIONALE,
    TEST_SEED,
    V2_BLEND,
    V2_DEPTH,
    study_cities,
    v2_features,
)
from evaluation.v3 import apply_interval, fit_holdout, select_v3

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
FIGURES = REPORTS / "figures"


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(len(arr))}


def _summary(rows: list[dict]) -> dict[str, float]:
    out = {}
    for metric in ("r2", "mae", "rmse", "mape", "median_ape"):
        stats = _mean_std([row[metric] for row in rows])
        out[f"{metric}_mean"] = stats["mean"]
        out[f"{metric}_std"] = stats["std"]
    out["n_seeds"] = len(rows)
    return out


def _tagged(rows: list[dict], tag: str, city: str | None = None) -> list[dict]:
    picked = [row for row in rows if row["tag"] == tag and (city is None or row["city"] == city)]
    if not picked:
        raise RuntimeError(f"no rows for {tag} {city}")
    return picked


def _fit_eval(
    frame, split, train_idx, valid_idx, columns, spatial_mode, library, depth, monotone=False
):
    X_train, y_log, X_valid, y_valid, _encoder = design_matrices(
        frame, train_idx, valid_idx, columns, spatial_mode
    )
    model = fit_boosted(
        library,
        params_for_depth(depth),
        X_train,
        y_log,
        split["groups"][train_idx],
        monotone=monotone,
    )
    pred = predict_dollars(model, X_valid, library)
    metrics = regression_metrics(y_valid, pred)
    metrics["n_trees"] = int(getattr(model, "_capscout_trees", None) or 0)
    return metrics, pred, y_valid


def _record(bucket, tag, city, seed, metrics, pred=None, y_true=None):
    bucket.append({"tag": tag, "city": city, "seed": int(seed), **metrics})
    return pred, y_true


def _learning_curve(
    frame,
    split,
    train_idx,
    valid_idx,
    columns,
    spatial_mode,
    library,
    depth,
    monotone,
    text_mode="none",
):
    """One development seed, refit on a shrinking set of training cells.

    The curve is jagged because a random subset of cells is not a smooth
    "more data" knob. It is a diagnostic, not a tuning tool, which is why
    it uses seed 0 only and is not read by the selection rules.
    """
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
            frame, subset, valid_idx, columns, spatial_mode, text_mode=text_mode
        )
        model = fit_boosted(
            library, params_for_depth(depth), X_train, y_log, groups[subset], monotone=monotone
        )
        rows.append(
            {
                "fraction": fraction,
                "n_train": int(len(subset)),
                "valid_mae": regression_metrics(y_valid, predict_dollars(model, X_valid, library))[
                    "mae"
                ],
                "train_mae": regression_metrics(
                    np.expm1(y_log), predict_dollars(model, X_train, library)
                )["mae"],
            }
        )
    return rows


def _shap_rows(model, X: pd.DataFrame, library: str) -> list[dict]:
    sample = X.sample(n=min(600, len(X)), random_state=TEST_SEED)
    values = shap_matrix(model, sample, library)
    means = np.abs(values).mean(axis=0)
    order = np.argsort(-means)
    return [
        {"feature": str(sample.columns[i]), "mean_abs_shap": float(means[i])} for i in order[:15]
    ]


def _blend_shap(models: dict, libraries: list[str], X: pd.DataFrame) -> list[dict]:
    """Average of each model's log-space contributions.

    The prediction itself is an average of dollars. These reasons are the
    average of the two explanations, which is what you can say out loud
    without pretending the blend has one tree.
    """
    sample = X.sample(n=min(600, len(X)), random_state=TEST_SEED)
    stacked = [shap_matrix(models[library], sample, library) for library in libraries]
    means = np.mean(np.abs(np.stack(stacked, axis=0)), axis=0).mean(axis=0)
    names, values = _collapse_wording(list(sample.columns), means)
    order = np.argsort(-np.asarray(values))
    return [{"feature": names[int(i)], "mean_abs_shap": float(values[int(i)])} for i in order[:15]]


def _collapse_wording(names: list[str], means: np.ndarray) -> tuple[list[str], list[float]]:
    """Sum the TF-IDF components into one wording row.

    A single component is not something you can say out loud. The sum of
    the mean absolute contributions is how much the block moves the
    prediction, which is the number the chart should show.
    """
    kept_names: list[str] = []
    kept_values: list[float] = []
    wording = 0.0
    seen = False
    for name, value in zip(names, means, strict=True):
        if str(name).startswith("text_svd_"):
            wording += float(value)
            seen = True
        else:
            kept_names.append(str(name))
            kept_values.append(float(value))
    if seen:
        kept_names.append("listing_wording")
        kept_values.append(wording)
    return kept_names, kept_values


def _residual_bins(y_true, pred) -> list[dict]:
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


def _coverage_bins(frame: pd.DataFrame) -> list[dict]:
    part = frame.copy()
    try:
        part["bin"] = pd.qcut(part["pred_gbm"], 8, duplicates="drop")
    except ValueError:
        return []
    rows = []
    for bin_label, group in part.groupby("bin", observed=True):
        covered = (group["y_true"] >= group["pred_lo"]) & (group["y_true"] <= group["pred_hi"])
        rows.append(
            {
                "predicted_bin": str(bin_label),
                "n": int(len(group)),
                "coverage": float(covered.mean()),
                "median_pred": float(group["pred_gbm"].median()),
                "median_width": float((group["pred_hi"] - group["pred_lo"]).median()),
            }
        )
    return rows


def _room_errors(frame, test_idx, pred) -> list[dict]:
    part = frame.iloc[test_idx][["room_type", "price_clean"]].copy()
    part["pred"] = pred
    rows = []
    for room, group in part.groupby("room_type"):
        rows.append(
            {
                "room_type": str(room),
                **regression_metrics(group["price_clean"].to_numpy(), group["pred"].to_numpy()),
            }
        )
    return rows


def _prepare(city_key: str, seeds):
    frame, stats = load_study_frame(city_key)
    split = locked_split(
        frame["latitude"].to_numpy(dtype=float),
        frame["longitude"].to_numpy(dtype=float),
        frame["host_id"].to_numpy(),
    )
    inners = [
        item
        for item in development_splits(
            split["groups"],
            split["development_idx"],
            frame["latitude"].to_numpy(dtype=float),
            frame["longitude"].to_numpy(dtype=float),
            frame["host_id"].to_numpy(),
            seeds,
        )
        if item[0] in seeds
    ]
    return {"frame": frame, "stats": stats, "split": split, "inners": inners}


def _pdp_for(
    model,
    library,
    models,
    blend_libraries,
    X: pd.DataFrame,
    encoder,
    neighbourhood,
    columns_present,
):
    """A few curves a person can talk through. Development rows only."""
    wanted = []
    for name in (
        "bedrooms",
        "bathrooms",
        "distance_to_center_km",
        "census_median_income",
        "osm_dist_beach_km",
        "osm_n_food_1km",
    ):
        if name in columns_present:
            wanted.append(name)
    wanted = wanted[:4]
    sample_n = min(400, len(X))
    take = X.sample(n=sample_n, random_state=0)
    neigh = neighbourhood[take.index.to_numpy()] if neighbourhood is not None else None
    curves = {}

    def _predict(matrix: pd.DataFrame) -> np.ndarray:
        if blend_libraries:
            stacked = [predict_dollars(models[name], matrix, name) for name in blend_libraries]
            return np.mean(stacked, axis=0)
        return predict_dollars(model, matrix, library)

    for name in wanted:
        if name == "bedrooms":
            grid = [0, 1, 2, 3, 4, 5]
        elif name == "bathrooms":
            grid = [1.0, 1.5, 2.0, 2.5, 3.0]
        elif name == "distance_to_center_km":
            grid = [1, 3, 5, 10, 20]
        elif name == "osm_dist_beach_km":
            grid = [0.5, 1, 2, 5, 10]
        elif name == "osm_n_food_1km":
            grid = [0, 5, 20, 50, 100]
        else:
            series = take[name].dropna()
            grid = [float(v) for v in np.quantile(series, [0.1, 0.3, 0.5, 0.7, 0.9])]
        rows = []
        for value in grid:
            edited = take.copy()
            edited[name] = value
            fold = getattr(encoder, "fold", encoder)
            if (
                name == "bedrooms"
                and fold is not None
                and hasattr(fold, "bedroom_interaction")
                and "neigh_bed_log_price" in edited.columns
            ):
                edited["neigh_bed_log_price"] = fold.bedroom_interaction(
                    neigh, np.full(len(edited), value)
                )
            rows.append({"value": float(value), "mean_pred": float(_predict(edited).mean())})
        curves[name] = rows
    return curves


def _city_radius(fold_residuals: list[np.ndarray]) -> tuple[float, float | None]:
    if len(fold_residuals) == 1:
        return conformal_radius(fold_residuals[0]), None
    coverages = []
    for index, residual in enumerate(fold_residuals):
        pooled = np.concatenate(
            [fold_residuals[j] for j in range(len(fold_residuals)) if j != index]
        )
        radius = conformal_radius(pooled)
        coverages.append(float(np.mean(residual <= radius)))
    pooled_all = np.concatenate(fold_residuals)
    return conformal_radius(pooled_all), float(np.mean(coverages))


def run(smoke: bool = False) -> tuple[dict, pd.DataFrame, dict]:
    """Choose on the selection folds, then score the two holdouts once."""
    cities = ("austin",) if smoke else study_cities()
    seeds = (0,) if smoke else (0, 1, 2, 3, 4)
    if smoke:
        import evaluation.fitting as fitting
        import evaluation.protocol as protocol

        protocol.MAX_TREES = 40
        fitting.MAX_TREES = 40

    # Freeze both id files for every study city, even during a smoke run.
    # A smoke run that wrote only Austin would make the next full run look
    # like the holdout had changed.
    print("Freezing holdouts (coordinates and seeds only)", flush=True)
    ensure_frozen_ids(study_cities())
    ensure_confirm_ids(study_cities())
    print("External data", flush=True)
    ensure_external(cities)

    prepared = {}
    print("Loading and splitting", flush=True)
    for city in cities:
        frame, stats = load_study_frame(city)
        lat = frame["latitude"].to_numpy(dtype=float)
        lon = frame["longitude"].to_numpy(dtype=float)
        hosts = frame["host_id"].to_numpy()
        split = locked_split(lat, lon, hosts)
        confirm = confirmation_split(lat, lon, hosts, split["development_idx"], split["groups"])
        inners = development_splits(split["groups"], confirm["pool_idx"], lat, lon, hosts, seeds)
        prepared[city] = {
            "frame": frame,
            "stats": stats,
            "split": split,
            "confirm": confirm,
            "folds": [(seed, train_idx, valid_idx) for seed, train_idx, valid_idx, _info in inners],
        }
        info = split["info"]
        print(
            f"  {city}: {stats['rows_final']} rows, locked test {info['n_test']}, "
            f"confirmation {confirm['info']['n_confirm']}, "
            f"selection pool {confirm['info']['n_pool']}",
            flush=True,
        )

    decision = select_v3(prepared, smoke=smoke)
    spec = {
        "columns": decision["columns"],
        "spatial_mode": decision["spatial_mode"],
        "text_mode": decision["text_mode"],
        "depth": decision["depth"],
        "pooled": decision["pooled"],
    }
    v2_columns, v2_spatial = v2_features()
    v2_spec = {
        "columns": v2_columns,
        "spatial_mode": v2_spatial,
        "text_mode": "none",
        "depth": V2_DEPTH,
        "pooled": False,
    }

    reference_rows: list[dict] = []
    curve_rows: dict[str, list] = {}
    print("Baselines and ridge on the selection folds", flush=True)
    for city, pack in prepared.items():
        frame = pack["frame"]
        y = frame["price_clean"].to_numpy(dtype=float)
        for seed, train_idx, valid_idx in pack["folds"]:
            reference_rows.append(
                {
                    "city": city,
                    "seed": seed,
                    "model": "baseline_room",
                    **regression_metrics(
                        y[valid_idx],
                        median_baseline(
                            frame["neighbourhood_cleansed"].to_numpy(),
                            frame["room_type"].to_numpy(),
                            frame["bedrooms"].to_numpy(dtype=float),
                            y,
                            train_idx,
                            valid_idx,
                            by_bedroom=False,
                        ),
                    ),
                }
            )
            reference_rows.append(
                {
                    "city": city,
                    "seed": seed,
                    "model": "baseline_room_bed",
                    **regression_metrics(
                        y[valid_idx],
                        median_baseline(
                            frame["neighbourhood_cleansed"].to_numpy(),
                            frame["room_type"].to_numpy(),
                            frame["bedrooms"].to_numpy(dtype=float),
                            y,
                            train_idx,
                            valid_idx,
                            by_bedroom=True,
                        ),
                    ),
                }
            )
            X_train, y_log, X_valid, y_valid, _state = design_matrices(
                frame,
                train_idx,
                valid_idx,
                spec["columns"],
                spec["spatial_mode"],
                text_mode=spec["text_mode"],
            )
            reference_rows.append(
                {
                    "city": city,
                    "seed": seed,
                    "model": "ridge",
                    **regression_metrics(y_valid, fit_predict_ridge(X_train, y_log, X_valid)),
                }
            )
        if not smoke and pack["folds"]:
            seed, train_idx, valid_idx = pack["folds"][0]
            curve_rows[city] = _learning_curve(
                frame,
                pack["split"],
                train_idx,
                valid_idx,
                spec["columns"],
                spec["spatial_mode"],
                "xgboost",
                spec["depth"],
                False,
                text_mode=spec["text_mode"],
            )

    print("Locked test, once", flush=True)
    train_idxs = {city: pack["split"]["development_idx"] for city, pack in prepared.items()}
    test_idxs = {city: pack["split"]["test_idx"] for city, pack in prepared.items()}
    fitted = fit_holdout(prepared, train_idxs, test_idxs, spec, refit=True)
    print("Published v2 specification, refit on the same rows", flush=True)
    v2_fitted = fit_holdout(prepared, train_idxs, test_idxs, v2_spec, refit=True)

    print("Confirmation holdout, once", flush=True)
    confirm_train = {city: pack["confirm"]["train_idx"] for city, pack in prepared.items()}
    confirm_eval = {city: pack["confirm"]["confirm_idx"] for city, pack in prepared.items()}
    confirm_fit = fit_holdout(prepared, confirm_train, confirm_eval, spec, refit=True)
    confirm_v2 = fit_holdout(prepared, confirm_train, confirm_eval, v2_spec, refit=True)

    interval_method = decision["interval_method"]
    predictions = []
    confirm_predictions = []
    city_reports = {}
    shap_report = {}
    pdp_report = {}
    for city, pack in prepared.items():
        frame = pack["frame"]
        split = pack["split"]
        dev = split["development_idx"]
        test = split["test_idx"]
        held = fitted[city]
        audit = {
            "split": split_audit(frame, dev, test),
            "columns": column_audit(held["columns"]),
            "confirmation": split_audit(
                frame, pack["confirm"]["train_idx"], pack["confirm"]["confirm_idx"]
            ),
        }
        if not audit["split"]["passed"] or not audit["columns"]["passed"]:
            raise RuntimeError(f"Audit failed for {city}: {audit}")
        if not audit["confirmation"]["passed"]:
            raise RuntimeError(f"Confirmation audit failed for {city}: {audit['confirmation']}")

        y_test = held["y"]
        pred = held["pred"]
        q_lo = held["q_lo"]
        q_hi = held["q_hi"]
        method_bounds = {}
        for method_name in ("log_conformal", "scaled_log", "cqr"):
            low_m, high_m = apply_interval(
                method_name, pred, q_lo, q_hi, decision["interval_final"][method_name][city]
            )
            method_bounds[method_name] = (low_m, high_m)
        low, high = method_bounds[interval_method]
        raw_low, raw_high = cqr_bounds(q_lo, q_hi, 0.0)

        groups = split["groups"][dev]
        prev_columns = list(SIZE_COLUMNS) + list(LOCATION_COLUMNS)
        X_prev, y_prev, X_prev_test, _y_prev, _enc_prev = design_matrices(
            frame, dev, test, prev_columns, "none"
        )
        prev_stopped = fit_boosted(
            "xgboost", params_for_depth(4), X_prev, y_prev, groups, monotone=False
        )
        prev_trees = int(getattr(prev_stopped, "_capscout_trees", None) or 80)
        prev_model = refit_boosted(
            "xgboost", params_for_depth(4), prev_trees, X_prev, y_prev, monotone=False
        )
        prev_pred = predict_dollars(prev_model, X_prev_test, "xgboost")

        X_dev, y_log, X_test_ridge, _y_ridge, _ridge_state = design_matrices(
            frame, dev, test, spec["columns"], spec["spatial_mode"], text_mode=spec["text_mode"]
        )
        ridge_pred = fit_predict_ridge(X_dev, y_log, X_test_ridge)
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
        chosen_candidate = next(
            item for item in decision["interval_candidates"] if item["name"] == interval_method
        )
        locked = {
            "gbm": regression_metrics(y_test, pred),
            "v2_refit": regression_metrics(y_test, v2_fitted[city]["pred"]),
            "previous_spec": regression_metrics(y_test, prev_pred),
            "ridge": regression_metrics(y_test, ridge_pred),
            "baseline_room": regression_metrics(y_test, base_pred),
            "baseline_room_bed": regression_metrics(y_test, base_bed_pred),
            "interval": {
                "method": interval_method,
                "nominal": INTERVAL_LEVEL,
                "development_coverage": chosen_candidate["by_city"][city]["coverage"],
                "development_upper_coverage": chosen_candidate["by_city"][city]["upper_coverage"],
                **interval_metrics(y_test, low, high),
            },
            "interval_methods": {
                name: interval_metrics(y_test, bounds[0], bounds[1])
                for name, bounds in method_bounds.items()
            },
            "quantile_interval": {
                "library": "xgboost",
                "alphas": [0.1, 0.9],
                "note": "Uncalibrated quantile band. The conformalized version is interval_methods.cqr.",
                **interval_metrics(y_test, raw_low, raw_high),
            },
        }
        print(
            f"  {city} test R2 {locked['gbm']['r2']:.3f} MAE {locked['gbm']['mae']:.2f} "
            f"coverage {locked['interval']['coverage']:.3f}",
            flush=True,
        )
        for i, row_idx in enumerate(test):
            predictions.append(
                {
                    "city": city,
                    "listing_id": str(int(frame.at[row_idx, "id"])),
                    "y_true": float(y_test[i]),
                    "pred_gbm": float(pred[i]),
                    "pred_lo": float(low[i]),
                    "pred_hi": float(high[i]),
                    "pred_q10": float(raw_low[i]),
                    "pred_q90": float(raw_high[i]),
                    "pred_v2_refit": float(v2_fitted[city]["pred"][i]),
                    "pred_previous": float(prev_pred[i]),
                    "pred_ridge": float(ridge_pred[i]),
                    "pred_baseline": float(base_pred[i]),
                    "pred_baseline_bed": float(base_bed_pred[i]),
                    "room_type": str(frame.at[row_idx, "room_type"]),
                    "neighbourhood": str(frame.at[row_idx, "neighbourhood_cleansed"]),
                    "distance_to_center_km": float(frame.at[row_idx, "distance_to_center_km"]),
                    "bedrooms": float(frame.at[row_idx, "bedrooms"]),
                }
            )
        confirm_y = confirm_fit[city]["y"]
        confirm_pred = confirm_fit[city]["pred"]
        c_low, c_high = apply_interval(
            interval_method,
            confirm_pred,
            confirm_fit[city]["q_lo"],
            confirm_fit[city]["q_hi"],
            decision["interval_final"][interval_method][city],
        )
        for i, row_idx in enumerate(pack["confirm"]["confirm_idx"]):
            confirm_predictions.append(
                {
                    "city": city,
                    "listing_id": str(int(frame.at[row_idx, "id"])),
                    "y_true": float(confirm_y[i]),
                    "pred_gbm": float(confirm_pred[i]),
                    "pred_lo": float(c_low[i]),
                    "pred_hi": float(c_high[i]),
                    "pred_v2_refit": float(confirm_v2[city]["pred"][i]),
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
        state = held["state"]
        artifact = {
            "model": None,
            "models": held["models"],
            "library": "blend",
            "blend_libraries": list(V2_BLEND),
            "depth": spec["depth"],
            "monotonic": False,
            "n_trees": held["tree_counts"],
            "encoder": state.fold,
            "kernel": state.kernel,
            "text_encoder": state.text,
            "columns": list(held["columns"]),
            "spatial_mode": spec["spatial_mode"],
            "text_mode": spec["text_mode"],
            "pooled": bool(spec["pooled"]),
            "feature_set": decision["feature_set"],
            "conformal_log_radius": float(
                decision["interval_final"]["log_conformal"][city]["radius"]
            ),
            "interval_method": interval_method,
            "interval_params": decision["interval_final"][interval_method][city],
            "quantile_models": held["q_models"] if interval_method == "cqr" else None,
            "interval_level": INTERVAL_LEVEL,
            "occupancy_by_neighbourhood": {str(k): float(v) for k, v in occ_map.items()},
            "occupancy_global": float(dev_occ["occupancy_rate"].median()) if len(dev_occ) else 0.5,
            "training_latitude": frame["latitude"].to_numpy(dtype=np.float32)[dev],
            "training_longitude": frame["longitude"].to_numpy(dtype=np.float32)[dev],
            "distance_p95": float(frame.iloc[dev]["distance_to_center_km"].quantile(0.95)),
        }
        if not smoke:
            save_artifact(city, artifact)
        shap_report[city] = _blend_shap(held["models"], list(V2_BLEND), held["matrix_valid"])
        X_pdp, _y_pdp, _x_ignore, _y_ignore, _pdp_state = design_matrices(
            frame,
            dev,
            dev[:1],
            spec["columns"],
            spec["spatial_mode"],
            text_mode=spec["text_mode"],
            text_encoder=state.text,
        )
        if spec["pooled"]:
            from features.columns import city_dummy

            for key in cities:
                X_pdp[city_dummy(key)] = 1.0 if key == city else 0.0
            X_pdp = X_pdp[held["columns"]]
        neighbourhood = frame["neighbourhood_cleansed"].to_numpy()[dev]
        pdp_report[city] = _pdp_for(
            None,
            "xgboost",
            held["models"],
            list(V2_BLEND),
            X_pdp,
            state,
            neighbourhood,
            list(X_pdp.columns),
        )
        confirm_metrics = {
            "gbm": regression_metrics(confirm_y, confirm_pred),
            "v2_refit": regression_metrics(confirm_y, confirm_v2[city]["pred"]),
            "interval": {
                "method": interval_method,
                "nominal": INTERVAL_LEVEL,
                **interval_metrics(confirm_y, c_low, c_high),
            },
            "n_train": int(len(pack["confirm"]["train_idx"])),
            **{
                key: pack["confirm"]["info"][key]
                for key in ("n_confirm", "removed_host", "removed_buffer", "train_after")
            },
        }
        city_reports[city] = {
            "display": CITIES[city]["display"],
            "snapshot": pack["stats"]["snapshot"],
            "cleaning": pack["stats"],
            "split": split["info"],
            "confirmation_split": pack["confirm"]["info"],
            "locked_test": locked,
            "confirmation": confirm_metrics,
            "residuals_by_predicted_quintile": _residual_bins(y_test, pred),
            "locked_test_by_room_type": _room_errors(frame, test, pred),
            "interval_by_predicted_bin": [],
            "shap": shap_report[city],
            "partial_dependence": pdp_report[city],
            "audit": audit,
        }

    prediction_frame = pd.DataFrame(predictions)
    confirm_frame = pd.DataFrame(confirm_predictions)
    for city in cities:
        city_reports[city]["interval_by_predicted_bin"] = _coverage_bins(
            prediction_frame[prediction_frame["city"] == city]
        )

    short_tags = {
        "v2": "v3:v2",
        "detail": "v3:alone:listing_detail",
        "lexicon": "v3:alone:text_lexicon",
        "wording": "v3:alone:text_svd",
        "kernel": "v3:alone:kernel_price",
        "selected": "v3:selected",
    }
    ablation = {}
    for city in cities:
        ablation[city] = {}
        for name, tag in short_tags.items():
            ablation[city][name] = _summary(_tagged(decision["rows"], tag, city))

    comparison = {}
    for city in cities:
        comparison[city] = {}
        for model_name in ("baseline_room", "baseline_room_bed", "ridge"):
            part = [
                row for row in reference_rows if row["city"] == city and row["model"] == model_name
            ]
            comparison[city][model_name] = _summary(part)
        comparison[city]["v2"] = _summary(_tagged(decision["rows"], "v3:v2", city))
        comparison[city]["gbm"] = _summary(_tagged(decision["rows"], "v3:selected", city))

    archive_path = REPORTS / "v2_metrics.json"
    published = None
    if archive_path.exists():
        archive = json.loads(archive_path.read_text())
        published = {
            city: {
                "gbm": archive["cities"][city]["locked_test"]["gbm"],
                "interval_coverage": archive["cities"][city]["locked_test"]["interval"]["coverage"],
                "interval_median_width": archive["cities"][city]["locked_test"]["interval"][
                    "median_width"
                ],
            }
            for city in archive["cities"]
            if city in cities
        }

    scored_columns = fitted[cities[0]]["columns"]
    result = {
        "study_cities": list(cities),
        "rationale": STUDY_RATIONALE,
        "protocol": {
            "test_seed": TEST_SEED,
            "confirm_seed": CONFIRM_SEED,
            "previous_study_seed": 42,
            "inner_seeds": list(seeds),
            "v2_feature_set": "plus_census+listing_quality",
            "v3_kept": decision["kept"],
            "v3_base_mae": decision["base_mae"],
            "v3_trials": decision["trials"],
            "alone_development_mae": decision["alone"],
            "chosen_depth": decision["depth"],
            "depth_development_mae": decision["depth_mae"],
            "pooled": decision["pooled"],
            "pooled_development_mae": decision["pooled_mae"],
            "per_city_development_mae": decision["per_city_mae"],
            "interval_method": interval_method,
            "interval_candidates": decision["interval_candidates"],
            "library": "blend",
            "blend_libraries": list(V2_BLEND),
            "monotonic": False,
            "spatial_mode": spec["spatial_mode"],
            "text_mode": spec["text_mode"],
            "feature_set": decision["feature_set"],
            "static_columns": decision["columns"],
            "columns": scored_columns,
            "feature_set_why": FEATURE_SET_WHY,
            "selection_rule": (
                "v2's published specification is the starting point and is not re-searched. "
                "Add-ons are tried in a fixed order on development folds that exclude the "
                "seed-11 confirmation cells. Each stays only if it cuts mean MAE by at least 1%. "
                "Depth is then re-checked, shallower within 1%. A pooled model replaces the "
                "per-city blend only if it is at least 1% better. The interval stays the "
                "log-residual band unless a challenger improves expensive-listing coverage "
                "on those same folds without getting much wider. The locked test is scored "
                "once after that, and it is not an input to any of those choices. "
                "The confirmation holdout is also scored once and is not an input."
            ),
        },
        "published_v2": published,
        "cities": city_reports,
        "development_ablation": ablation,
        "development_comparison": comparison,
        "learning_curves": curve_rows,
        "partial_dependence": pdp_report,
        "leakage_demo": (
            austin_leakage_demo(prepared["austin"]["frame"]) if "austin" in prepared else None
        ),
        "audit_passed": all(
            report["audit"]["split"]["passed"]
            and report["audit"]["columns"]["passed"]
            and report["audit"]["confirmation"]["passed"]
            for report in city_reports.values()
        ),
        "smoke": smoke,
        "feature_labels": {name: pretty(name) for name in scored_columns},
        "confirmation_note": (
            f"Seed {CONFIRM_SEED} holds out development cells that were inside v2's "
            "cross-validation, so it is not a blind test of the v2 specification. "
            "It is a blind test of every choice made in this round. Seed 7 was published "
            "with v2, so this round does not use it to choose anything. It is scored once, "
            "on the same rows, with the new model trained on the same development listings v2 used."
        ),
    }
    if not smoke:
        confirm_frame.to_csv(REPORTS / "confirmation_predictions.csv", index=False)
    return result, prediction_frame, shap_report


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
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate the nightly-price study.")
    parser.add_argument("--smoke", action="store_true", help="Austin only, one seed, tiny trees.")
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
    plot_model_comparison(
        result["development_comparison"],
        FIGURES / "model_comparison.png",
        ["baseline_room", "ridge", "v2", "gbm"],
    )
    if result["learning_curves"]:
        plot_learning_curves(result["learning_curves"], FIGURES / "learning_curves.png")
    plot_pred_vs_actual(predictions, FIGURES / "pred_vs_actual.png")
    plot_residuals(predictions, FIGURES / "residuals.png")
    plot_shap(shap_report, FIGURES / "shap.png")
    plot_calibration(predictions, FIGURES / "calibration.png")
    plot_pdp(result["partial_dependence"], FIGURES / "partial_dependence.png")
    print("Wrote reports/metrics.json, locked_test_predictions.csv, and figures/", flush=True)


if __name__ == "__main__":
    main()
