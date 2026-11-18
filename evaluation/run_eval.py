"""Run the study and write reports/metrics.json plus the figures.

    python -m evaluation.run_eval
    python -m evaluation.run_eval --smoke

The locked test is not read until `score_locked_test`. Feature sets,
depth, library, monotone constraints, and the blend are chosen from
development splits only. `--smoke` is a crash check, not a result.
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
    KNN_COLUMNS,
    LOCATION_COLUMNS,
    QUALITY_COLUMNS,
    SIZE_COLUMNS,
    SMOOTH_COLUMNS,
    pretty,
)
from models.artifacts import save_artifact
from models.explain import shap_matrix

from evaluation.audit import column_audit, split_audit
from evaluation.blocking import development_splits, locked_split
from evaluation.fitting import (
    design_matrices,
    fit_boosted,
    fit_predict_ridge,
    fit_quantile,
    median_baseline,
    params_for_depth,
    predict_dollars,
    refit_boosted,
)
from evaluation.freeze import ensure_frozen_ids
from evaluation.intervals import apply_log_interval, conformal_radius, log_abs_residual
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
    DEPTHS,
    FEATURE_SET_WHY,
    INTERVAL_LEVEL,
    LIBRARIES,
    OSM_WITHOUT_CENSUS,
    STAGE1,
    STUDY_RATIONALE,
    TEST_SEED,
    resolve_features,
    study_cities,
)
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
    frame, split, train_idx, valid_idx, columns, spatial_mode, library, depth, monotone
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
            frame, subset, valid_idx, columns, spatial_mode
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
    order = np.argsort(-means)
    return [
        {"feature": str(sample.columns[i]), "mean_abs_shap": float(means[i])} for i in order[:15]
    ]


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
            if (
                name == "bedrooms"
                and encoder is not None
                and "neigh_bed_log_price" in edited.columns
            ):
                edited["neigh_bed_log_price"] = encoder.bedroom_interaction(
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
    cities = ("austin",) if smoke else study_cities()
    seeds = (0,) if smoke else (0, 1, 2, 3, 4)
    if smoke:
        import evaluation.fitting as fitting
        import evaluation.protocol as protocol

        protocol.MAX_TREES = 40
        fitting.MAX_TREES = 40

    print("Freezing the locked test (coordinates and seed only)", flush=True)
    ensure_frozen_ids(cities)
    print("External data", flush=True)
    ensure_external(cities)

    prepared = {}
    print("Loading and splitting", flush=True)
    for city in cities:
        prepared[city] = _prepare(city, seeds)
        info = prepared[city]["split"]["info"]
        stats = prepared[city]["stats"]
        print(
            f"  {city}: {stats['rows_final']} rows, test {info['n_test']}, "
            f"dev {info['train_after']} "
            f"(dropped {info['removed_host']} host, {info['removed_buffer']} buffer), "
            f"tract polygon {stats['place']['census_polygon_fraction']:.1%}",
            flush=True,
        )

    rows: list[dict] = []
    library_preds: dict[tuple, tuple] = {}
    mono_preds: dict[tuple, tuple] = {}

    stage1 = STAGE1 if not smoke else STAGE1[:2]
    print("Stage 1 ablation, XGBoost depth 6", flush=True)
    diagnostic = [OSM_WITHOUT_CENSUS] if not smoke else []
    for name, columns, mode in list(stage1) + diagnostic:
        for city, pack in prepared.items():
            for seed, train_idx, valid_idx, _info in pack["inners"]:
                metrics, _pred, _y = _fit_eval(
                    pack["frame"], pack["split"], train_idx, valid_idx, columns, mode, "xgboost", 6
                )
                _record(rows, f"feature:{name}", city, seed, metrics)
                print(f"  {city} {name} seed {seed}: MAE {metrics['mae']:.2f}", flush=True)

    stage1_means = [
        (name, _summary(_tagged(rows, f"feature:{name}"))["mae_mean"]) for name, _c, _m in stage1
    ]
    stage1_name = choose_simpler(stage1_means)
    base_columns, _mode = resolve_features(stage1_name, None, False)
    base_mae = dict(stage1_means)[stage1_name]
    print(f"Stage 1: {stage1_name} MAE {base_mae:.2f}", flush=True)

    addon = None
    quality = False
    if not smoke:
        print("Stage 2 target encodings, on the stage-1 columns", flush=True)
        for addon_name, mode in (("target_smooth", "smooth"), ("target_knn", "knn")):
            for city, pack in prepared.items():
                for seed, train_idx, valid_idx, _info in pack["inners"]:
                    metrics, _pred, _y = _fit_eval(
                        pack["frame"],
                        pack["split"],
                        train_idx,
                        valid_idx,
                        base_columns,
                        mode,
                        "xgboost",
                        6,
                    )
                    _record(rows, f"feature:{addon_name}", city, seed, metrics)
                    print(
                        f"  {city} {addon_name} seed {seed}: MAE {metrics['mae']:.2f}", flush=True
                    )
        addon_means = [
            (name, _summary(_tagged(rows, f"feature:{name}"))["mae_mean"])
            for name in ("target_smooth", "target_knn")
        ]
        addon = choose_addon(base_mae, addon_means)
        columns, spatial_mode = resolve_features(stage1_name, addon, False)
        current_mae = dict(addon_means)[addon] if addon else base_mae
        print("Stage 3 listing quality", flush=True)
        for city, pack in prepared.items():
            for seed, train_idx, valid_idx, _info in pack["inners"]:
                metrics, _pred, _y = _fit_eval(
                    pack["frame"],
                    pack["split"],
                    train_idx,
                    valid_idx,
                    columns + list(QUALITY_COLUMNS),
                    spatial_mode,
                    "xgboost",
                    6,
                )
                _record(rows, "feature:listing_quality", city, seed, metrics)
                print(f"  {city} listing quality seed {seed}: MAE {metrics['mae']:.2f}", flush=True)
        quality_mae = _summary(_tagged(rows, "feature:listing_quality"))["mae_mean"]
        quality = choose_quality(current_mae, quality_mae)
    columns, spatial_mode = resolve_features(stage1_name, addon, quality)
    chosen_set = feature_label(stage1_name, addon, quality)
    print(f"Chosen features: {chosen_set}", flush=True)

    depths = (4,) if smoke else DEPTHS
    print("Depth search", flush=True)
    for depth in depths:
        for city, pack in prepared.items():
            for seed, train_idx, valid_idx, _info in pack["inners"]:
                metrics, _pred, _y = _fit_eval(
                    pack["frame"],
                    pack["split"],
                    train_idx,
                    valid_idx,
                    columns,
                    spatial_mode,
                    "xgboost",
                    depth,
                )
                _record(rows, f"depth:{depth}", city, seed, metrics)
                print(
                    f"  {city} xgb depth {depth} seed {seed}: MAE {metrics['mae']:.2f}", flush=True
                )
    chosen_depth = choose_depth(
        [(depth, _summary(_tagged(rows, f"depth:{depth}"))["mae_mean"]) for depth in depths]
    )
    print(f"Chosen depth: {chosen_depth}", flush=True)

    libraries = ("xgboost",) if smoke else LIBRARIES
    print("Libraries", flush=True)
    for library in libraries:
        for city, pack in prepared.items():
            for seed, train_idx, valid_idx, _info in pack["inners"]:
                metrics, pred, y_true = _fit_eval(
                    pack["frame"],
                    pack["split"],
                    train_idx,
                    valid_idx,
                    columns,
                    spatial_mode,
                    library,
                    chosen_depth,
                )
                _record(rows, f"library:{library}", city, seed, metrics)
                library_preds[(city, seed, library)] = (y_true, pred)
                print(f"  {city} {library} seed {seed}: MAE {metrics['mae']:.2f}", flush=True)
    library_scores = {
        library: _summary(_tagged(rows, f"library:{library}"))["mae_mean"] for library in libraries
    }
    if smoke:
        library_scores.setdefault("lightgbm", library_scores["xgboost"] * 2)
        library_scores.setdefault("catboost", library_scores["xgboost"] * 2)
    chosen_library = choose_library(library_scores)
    print(f"Chosen library: {chosen_library}", flush=True)

    use_mono = False
    if not smoke:
        print("Monotone constraints", flush=True)
        for city, pack in prepared.items():
            for seed, train_idx, valid_idx, _info in pack["inners"]:
                metrics, pred, y_true = _fit_eval(
                    pack["frame"],
                    pack["split"],
                    train_idx,
                    valid_idx,
                    columns,
                    spatial_mode,
                    chosen_library,
                    chosen_depth,
                    monotone=True,
                )
                _record(rows, "monotonic", city, seed, metrics)
                mono_preds[(city, seed)] = (y_true, pred)
                print(f"  {city} monotone seed {seed}: MAE {metrics['mae']:.2f}", flush=True)
        mono_mae = _summary(_tagged(rows, "monotonic"))["mae_mean"]
        use_mono = choose_monotonic(library_scores[chosen_library], mono_mae)
    else:
        mono_mae = None
    print(f"Monotone: {use_mono}", flush=True)

    use_blend = False
    blend_pair: list[str] = []
    blend_mae = None
    if not smoke:
        ranked = sorted(library_scores, key=library_scores.get)
        blend_pair = list(ranked[:2])
        blend_values = []
        for city, pack in prepared.items():
            for seed, _train, _valid, _info in pack["inners"]:
                y_a, pred_a = library_preds[(city, seed, blend_pair[0])]
                _y_b, pred_b = library_preds[(city, seed, blend_pair[1])]
                blended = 0.5 * (pred_a + pred_b)
                metrics = regression_metrics(y_a, blended)
                _record(rows, "blend", city, seed, metrics)
                blend_values.append(metrics["mae"])
        blend_mae = float(np.mean(blend_values))
        winner_mae = mono_mae if use_mono else library_scores[chosen_library]
        use_blend = choose_blend(winner_mae, blend_mae)
    print(f"Blend {blend_pair}: {use_blend} MAE {blend_mae}", flush=True)

    point_library = "blend" if use_blend else chosen_library
    spatial_columns = list(columns)
    if spatial_mode == "smooth":
        spatial_columns += list(SMOOTH_COLUMNS)
    elif spatial_mode == "knn":
        spatial_columns += list(KNN_COLUMNS)

    reference_rows: list[dict] = []
    curve_rows: dict[str, list] = {}
    print("Baselines, ridge, learning curves", flush=True)
    for city, pack in prepared.items():
        for seed, train_idx, valid_idx, _info in pack["inners"]:
            y = pack["frame"]["price_clean"].to_numpy(dtype=float)
            reference_rows.append(
                {
                    "city": city,
                    "seed": seed,
                    "model": "baseline_room",
                    **regression_metrics(
                        y[valid_idx],
                        median_baseline(
                            pack["frame"]["neighbourhood_cleansed"].to_numpy(),
                            pack["frame"]["room_type"].to_numpy(),
                            pack["frame"]["bedrooms"].to_numpy(dtype=float),
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
                            pack["frame"]["neighbourhood_cleansed"].to_numpy(),
                            pack["frame"]["room_type"].to_numpy(),
                            pack["frame"]["bedrooms"].to_numpy(dtype=float),
                            y,
                            train_idx,
                            valid_idx,
                            by_bedroom=True,
                        ),
                    ),
                }
            )
            X_train, y_log, X_valid, y_valid, _encoder = design_matrices(
                pack["frame"], train_idx, valid_idx, columns, spatial_mode
            )
            reference_rows.append(
                {
                    "city": city,
                    "seed": seed,
                    "model": "ridge",
                    **regression_metrics(y_valid, fit_predict_ridge(X_train, y_log, X_valid)),
                }
            )
        if not smoke:
            seed, train_idx, valid_idx, _info = pack["inners"][0]
            curve_rows[city] = _learning_curve(
                pack["frame"],
                pack["split"],
                train_idx,
                valid_idx,
                columns,
                spatial_mode,
                chosen_library,
                chosen_depth,
                use_mono and not use_blend,
            )

    radii = {}
    dev_coverage = {}
    for city in cities:
        folds = []
        for seed, _train, _valid, _info in prepared[city]["inners"]:
            if use_blend:
                y_a, pred_a = library_preds[(city, seed, blend_pair[0])]
                _y_b, pred_b = library_preds[(city, seed, blend_pair[1])]
                y_true, pred = y_a, 0.5 * (pred_a + pred_b)
            elif use_mono:
                y_true, pred = mono_preds[(city, seed)]
            else:
                y_true, pred = library_preds[(city, seed, chosen_library)]
            folds.append(log_abs_residual(y_true, pred))
        radii[city], dev_coverage[city] = _city_radius(folds)
        print(
            f"  {city} conformal radius {radii[city]:.3f} dev coverage {dev_coverage[city]}",
            flush=True,
        )

    print("Locked test, once", flush=True)
    predictions = []
    city_reports = {}
    shap_report = {}
    pdp_report = {}
    audits = {}
    for city, pack in prepared.items():
        frame = pack["frame"]
        split = pack["split"]
        dev = split["development_idx"]
        test = split["test_idx"]
        audits[city] = {
            "split": split_audit(frame, dev, test),
            "columns": column_audit(spatial_columns),
        }
        if not audits[city]["split"]["passed"] or not audits[city]["columns"]["passed"]:
            raise RuntimeError(f"Audit failed for {city}: {audits[city]}")

        X_dev, y_log, X_test, y_test, encoder = design_matrices(
            frame, dev, test, columns, spatial_mode
        )
        groups = split["groups"][dev]
        if use_blend:
            models = {}
            tree_counts = {}
            preds = []
            for library in blend_pair:
                stopped = fit_boosted(
                    library, params_for_depth(chosen_depth), X_dev, y_log, groups, monotone=False
                )
                n_trees = int(getattr(stopped, "_capscout_trees", None) or 80)
                model = refit_boosted(
                    library, params_for_depth(chosen_depth), n_trees, X_dev, y_log, monotone=False
                )
                models[library] = model
                tree_counts[library] = n_trees
                preds.append(predict_dollars(model, X_test, library))
            gbm_pred = np.mean(preds, axis=0)
            point_model = None
            saved_library = "blend"
        else:
            stopped = fit_boosted(
                chosen_library,
                params_for_depth(chosen_depth),
                X_dev,
                y_log,
                groups,
                monotone=use_mono,
            )
            n_trees = int(getattr(stopped, "_capscout_trees", None) or 80)
            point_model = refit_boosted(
                chosen_library,
                params_for_depth(chosen_depth),
                n_trees,
                X_dev,
                y_log,
                monotone=use_mono,
            )
            gbm_pred = predict_dollars(point_model, X_test, chosen_library)
            models = {chosen_library: point_model}
            tree_counts = {chosen_library: n_trees}
            saved_library = chosen_library

        # The previous study's specification, scored on this new holdout.
        # Depth 4, size and location, XGBoost, no new sources. It is a baseline,
        # not a candidate: nothing above was chosen by looking at this score.
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
        low, high = apply_log_interval(gbm_pred, radii[city])
        q_library = blend_pair[0] if use_blend else chosen_library
        q_trees = tree_counts[q_library]
        if smoke:
            q10 = q90 = gbm_pred
            q_cross = 0.0
        else:
            q10_model = fit_quantile(
                q_library, params_for_depth(chosen_depth), q_trees, X_dev, y_log, 0.1
            )
            q90_model = fit_quantile(
                q_library, params_for_depth(chosen_depth), q_trees, X_dev, y_log, 0.9
            )
            q10 = predict_dollars(q10_model, X_test, q_library)
            q90 = predict_dollars(q90_model, X_test, q_library)
            q_cross = float(np.mean(q10 > q90))
        q_low = np.minimum(q10, q90)
        q_high = np.maximum(q10, q90)

        locked = {
            "gbm": regression_metrics(y_test, gbm_pred),
            "previous_spec": regression_metrics(y_test, prev_pred),
            "ridge": regression_metrics(y_test, ridge_pred),
            "baseline_room": regression_metrics(y_test, base_pred),
            "baseline_room_bed": regression_metrics(y_test, base_bed_pred),
            "interval": {
                "method": "log_residual_conformal",
                "nominal": INTERVAL_LEVEL,
                "radius_log": radii[city],
                "development_coverage": dev_coverage[city],
                **interval_metrics(y_test, low, high),
            },
            "quantile_interval": {
                "library": q_library,
                "alphas": [0.1, 0.9],
                "cross_rate": q_cross,
                **interval_metrics(y_test, q_low, q_high),
            },
        }
        print(
            f"  {city} test {saved_library} R2 {locked['gbm']['r2']:.3f} "
            f"MAE {locked['gbm']['mae']:.2f} coverage {locked['interval']['coverage']:.3f}",
            flush=True,
        )
        for i, row_idx in enumerate(test):
            predictions.append(
                {
                    "city": city,
                    "listing_id": str(int(frame.at[row_idx, "id"])),
                    "y_true": float(y_test[i]),
                    "pred_gbm": float(gbm_pred[i]),
                    "pred_lo": float(low[i]),
                    "pred_hi": float(high[i]),
                    "pred_q10": float(q_low[i]),
                    "pred_q90": float(q_high[i]),
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
        dev_occ = frame.iloc[dev]
        dev_occ = dev_occ[np.isfinite(dev_occ["occupancy_rate"])]
        occ_map = (
            dev_occ.groupby("neighbourhood_cleansed")["occupancy_rate"]
            .median()
            .astype(float)
            .to_dict()
        )
        artifact = {
            "model": point_model,
            "models": models if use_blend else None,
            "library": saved_library,
            "blend_libraries": blend_pair if use_blend else None,
            "depth": chosen_depth,
            "monotonic": bool(use_mono and not use_blend),
            "n_trees": tree_counts,
            "encoder": encoder,
            "columns": list(columns),
            "spatial_mode": spatial_mode,
            "feature_set": chosen_set,
            "conformal_log_radius": float(radii[city]),
            "interval_level": INTERVAL_LEVEL,
            "occupancy_by_neighbourhood": {str(k): float(v) for k, v in occ_map.items()},
            "occupancy_global": float(dev_occ["occupancy_rate"].median()) if len(dev_occ) else 0.5,
            "training_latitude": frame["latitude"].to_numpy(dtype=np.float32)[dev],
            "training_longitude": frame["longitude"].to_numpy(dtype=np.float32)[dev],
            "distance_p95": float(frame.iloc[dev]["distance_to_center_km"].quantile(0.95)),
        }
        if not smoke:
            save_artifact(city, artifact)
        if use_blend:
            shap_report[city] = _blend_shap(models, blend_pair, X_test)
        else:
            shap_report[city] = _shap_rows(point_model, X_test, chosen_library)
        neighbourhood = frame["neighbourhood_cleansed"].to_numpy()[dev]
        X_dev_pdp = X_dev.copy()
        pdp_report[city] = _pdp_for(
            point_model,
            chosen_library,
            models,
            blend_pair if use_blend else None,
            X_dev_pdp,
            encoder,
            neighbourhood,
            list(X_dev.columns),
        )
        city_reports[city] = {
            "display": CITIES[city]["display"],
            "snapshot": pack["stats"]["snapshot"],
            "cleaning": pack["stats"],
            "split": split["info"],
            "locked_test": locked,
            "residuals_by_predicted_quintile": _residual_bins(y_test, gbm_pred),
            "locked_test_by_room_type": _room_errors(frame, test, gbm_pred),
            "interval_by_predicted_bin": [],
            "shap": shap_report[city],
            "partial_dependence": pdp_report[city],
            "audit": audits[city],
        }

    prediction_frame = pd.DataFrame(predictions)
    for city in cities:
        city_reports[city]["interval_by_predicted_bin"] = _coverage_bins(
            prediction_frame[prediction_frame["city"] == city]
        )

    ablation = {}
    feature_tags = [name for name, _c, _m in stage1]
    if not smoke:
        feature_tags += ["osm_without_census", "target_smooth", "target_knn", "listing_quality"]
    for city in cities:
        ablation[city] = {}
        for name in feature_tags:
            ablation[city][name] = _summary(_tagged(rows, f"feature:{name}", city))

    if use_blend:
        selected_tag = "blend"
    elif use_mono:
        selected_tag = "monotonic"
    else:
        selected_tag = f"library:{chosen_library}"

    comparison = {}
    for city in cities:
        comparison[city] = {}
        for model_name in ("baseline_room", "baseline_room_bed", "ridge"):
            part = [
                row for row in reference_rows if row["city"] == city and row["model"] == model_name
            ]
            comparison[city][model_name] = _summary(part)
        for library in libraries:
            comparison[city][library] = _summary(_tagged(rows, f"library:{library}", city))
        comparison[city]["gbm"] = _summary(_tagged(rows, selected_tag, city))

    leakage = austin_leakage_demo(prepared["austin"]["frame"]) if "austin" in prepared else None
    result = {
        "study_cities": list(cities),
        "rationale": STUDY_RATIONALE,
        "protocol": {
            "test_seed": TEST_SEED,
            "previous_study_seed": 42,
            "inner_seeds": list(seeds),
            "stage1": stage1_name,
            "addon": addon,
            "listing_quality": quality,
            "chosen_feature_set": chosen_set,
            "feature_set_why": FEATURE_SET_WHY,
            "stage1_development_mae": {name: mae for name, mae in stage1_means},
            "feature_set_development_mae": {
                name: _summary(_tagged(rows, f"feature:{name}"))["mae_mean"]
                for name in feature_tags
            },
            "chosen_depth": chosen_depth,
            "depth_development_mae": {
                str(depth): _summary(_tagged(rows, f"depth:{depth}"))["mae_mean"]
                for depth in depths
            },
            "library_development_mae": library_scores,
            "xgboost_development_mae": library_scores.get("xgboost"),
            "lightgbm_development_mae": library_scores.get("lightgbm"),
            "catboost_development_mae": library_scores.get("catboost"),
            "library": point_library,
            "component_library": chosen_library,
            "monotonic_within_bar": bool(use_mono),
            "monotonic": bool(use_mono and not use_blend),
            "monotonic_development_mae": mono_mae,
            "blend": bool(use_blend),
            "blend_libraries": blend_pair,
            "blend_development_mae": blend_mae,
            "spatial_mode": spatial_mode,
            "columns": spatial_columns,
            "ablation_depth": 6,
            "selection_rule": (
                "Stage 1 keeps the simplest cumulative prefix within 1% of the best. "
                "A target encoding or listing-quality block is added only if it improves "
                "development MAE by at least 1%. Shallower trees win within 1%. "
                "LightGBM or CatBoost replaces XGBoost only if it is at least 2% better. "
                "Monotone constraints are kept when they are within 1%. "
                "An equal-weight blend of the two best libraries is kept only if it is at least 1% better. "
                "osm_without_census is reported and is not eligible."
            ),
        },
        "cities": city_reports,
        "development_ablation": ablation,
        "development_comparison": comparison,
        "learning_curves": curve_rows,
        "partial_dependence": pdp_report,
        "leakage_demo": leakage,
        "audit_passed": all(
            report["audit"]["split"]["passed"] and report["audit"]["columns"]["passed"]
            for report in city_reports.values()
        ),
        "smoke": smoke,
        "feature_labels": {name: pretty(name) for name in spatial_columns},
    }
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
        ["baseline_room", "ridge", "xgboost", "lightgbm", "catboost", "gbm"],
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
