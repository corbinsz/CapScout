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
from data.history import ensure_history
from data.ingest import load_study_frame
from features.columns import (
    LOCATION_COLUMNS,
    SIZE_COLUMNS,
    pretty,
)
from models.artifacts import save_artifact
from models.explain import shap_matrix

from evaluation.audit import column_audit, split_audit
from evaluation.blocking import (
    confirmation_split,
    development_splits,
    final_split,
    locked_split,
    score_split,
)
from evaluation.confirm import ensure_confirm_ids
from evaluation.final import ensure_final_ids
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
from evaluation.intervals import conformal_radius
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
    FINAL_SEED,
    INTERVAL_LEVEL,
    STUDY_RATIONALE,
    TEST_SEED,
    V2_DEPTH,
    study_cities,
    v3_features,
)
from evaluation.splits import spatial_group_ids
from evaluation.v3 import fit_holdout as fit_v3_holdout
from evaluation.v4 import (
    apply_correction,
    apply_interval,
    build_matrices,
    fit_holdout,
    select_v4,
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
        if str(name).startswith("text_svd_") or str(name).startswith("text_embed_"):
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


def run(smoke: bool = False):
    """Choose on the selection folds, then score the final holdout once.

    Seed 7 and seed 11 are scored after that, as secondary comparisons.
    They are not inputs to the choices.
    """
    cities = ("austin",) if smoke else study_cities()
    seeds = (0,) if smoke else (0, 1, 2, 3, 4)
    if smoke:
        import evaluation.fitting as fitting
        import evaluation.protocol as protocol

        protocol.MAX_TREES = 40
        fitting.MAX_TREES = 40

    # Freeze every id file for every study city, even during a smoke run.
    # A smoke run that wrote only Austin would make the next full run look
    # like a holdout had changed. Prices are not read to do this.
    print("Freezing holdouts (coordinates and seeds only)", flush=True)
    ensure_frozen_ids(study_cities())
    ensure_confirm_ids(study_cities())
    ensure_final_ids(study_cities())
    print("External data", flush=True)
    ensure_external(cities)
    print("Earlier snapshots", flush=True)
    history = ensure_history(cities)

    v3_columns, v3_spatial, v3_text = v3_features()
    v3_spec = {
        "columns": v3_columns,
        "spatial_mode": v3_spatial,
        "text_mode": v3_text,
        "depth": V2_DEPTH,
        "pooled": False,
    }

    prepared = {}
    print("Loading and splitting", flush=True)
    for city in cities:
        frame, stats = load_study_frame(city)
        frame["months_before"] = 0.0
        lat = frame["latitude"].to_numpy(dtype=float)
        lon = frame["longitude"].to_numpy(dtype=float)
        hosts = frame["host_id"].to_numpy()
        split = locked_split(lat, lon, hosts)
        confirm = confirmation_split(lat, lon, hosts, split["development_idx"], split["groups"])
        final = final_split(lat, lon, hosts, confirm["pool_idx"], split["groups"])
        inners = development_splits(split["groups"], final["selection_idx"], lat, lon, hosts, seeds)
        prepared[city] = {
            "frame": frame,
            "extra": history[city],
            "stats": stats,
            "split": split,
            "confirm": confirm,
            "final": final,
            "blocked_idx": final["final_idx"],
            "folds": [(seed, train_idx, valid_idx) for seed, train_idx, valid_idx, _info in inners],
        }
        print(
            f"  {city}: {stats['rows_final']} rows, "
            f"final holdout {final['info']['n_final']}, "
            f"selection {final['info']['n_selection']}, "
            f"locked test {split['info']['n_test']}, "
            f"confirmation {confirm['info']['n_confirm']}, "
            f"extra listings {len(history[city])}",
            flush=True,
        )

    decision = select_v4(prepared, smoke=smoke)
    spec = decision["spec"]

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
            X_train, y_log, X_valid, y_valid, _weights, _groups, _state = build_matrices(
                frame, pack["extra"], train_idx, valid_idx, pack["blocked_idx"], spec
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
                v3_columns,
                "none",
                "xgboost",
                V2_DEPTH,
                False,
            )

    def _v3_predict(train_idxs, eval_idxs):
        fitted = fit_v3_holdout(
            prepared, train_idxs, eval_idxs, v3_spec, refit=True, with_quantiles=False
        )
        return {city: fitted[city]["pred"] for city in cities}

    def _v4_predict(train_idxs, eval_idxs):
        out = {}
        for city in cities:
            weights = None
            if decision["stacker"].get("kept"):
                weights = decision["stacker"]["weights"][city]
            fitted = fit_holdout(
                prepared[city]["frame"],
                prepared[city]["extra"],
                train_idxs[city],
                eval_idxs[city],
                spec,
                weights,
            )
            pred = apply_correction(
                fitted["pred"], decision["correction"], decision["correction_params"][city]
            )
            fitted["pred"] = pred
            out[city] = fitted
        return out

    print("Final holdout, once", flush=True)
    final_train = {}
    final_eval = {}
    for city, pack in prepared.items():
        lat = pack["frame"]["latitude"].to_numpy(dtype=float)
        lon = pack["frame"]["longitude"].to_numpy(dtype=float)
        hosts = pack["frame"]["host_id"].to_numpy()
        train_idx, info = score_split(lat, lon, hosts, pack["final"]["final_idx"])
        pack["final_score_train"] = train_idx
        pack["final_score_info"] = info
        final_train[city] = train_idx
        final_eval[city] = pack["final"]["final_idx"]
    final_fit = _v4_predict(final_train, final_eval)
    print("v3 specification on the final holdout", flush=True)
    final_v3 = _v3_predict(final_train, final_eval)

    print("Seed 7, once, as a secondary comparison", flush=True)
    locked_train = {city: pack["split"]["development_idx"] for city, pack in prepared.items()}
    locked_eval = {city: pack["split"]["test_idx"] for city, pack in prepared.items()}
    locked_fit = _v4_predict(locked_train, locked_eval)
    locked_v3 = _v3_predict(locked_train, locked_eval)

    print("Seed 11, once, as a secondary comparison", flush=True)
    confirm_train = {city: pack["confirm"]["train_idx"] for city, pack in prepared.items()}
    confirm_eval = {city: pack["confirm"]["confirm_idx"] for city, pack in prepared.items()}
    confirm_fit = _v4_predict(confirm_train, confirm_eval)
    confirm_v3 = _v3_predict(confirm_train, confirm_eval)

    interval_method = decision["interval_method"]
    final_rows = []
    locked_rows = []
    confirm_rows = []
    city_reports = {}
    shap_report = {}
    pdp_report = {}

    def _bounds(method, pred, city):
        return apply_interval(method, pred, decision["interval_final"][method][city])

    for city, pack in prepared.items():
        frame = pack["frame"]
        final_idx = pack["final"]["final_idx"]
        test_idx = pack["split"]["test_idx"]
        held = final_fit[city]
        audit = {
            "final": split_audit(frame, pack["final_score_train"], final_idx),
            "split": split_audit(frame, pack["split"]["development_idx"], test_idx),
            "columns": column_audit(held["columns"]),
            "confirmation": split_audit(
                frame, pack["confirm"]["train_idx"], pack["confirm"]["confirm_idx"]
            ),
        }
        for name in ("final", "split", "confirmation"):
            if not audit[name]["passed"]:
                raise RuntimeError(f"{name} audit failed for {city}: {audit[name]}")
        if not audit["columns"]["passed"]:
            raise RuntimeError(f"Column audit failed for {city}: {audit['columns']}")

        y = frame["price_clean"].to_numpy(dtype=float)
        pred = held["pred"]
        y_final = held["y"]
        method_bounds = {
            name: _bounds(name, pred, city)
            for name in ("scaled_log", "asymmetric", "mondrian_asymmetric")
        }
        low, high = method_bounds[interval_method]

        dev = pack["final_score_train"]
        prev_columns = list(SIZE_COLUMNS) + list(LOCATION_COLUMNS)
        X_prev, y_prev, X_prev_test, _y_prev, _enc_prev = design_matrices(
            frame, dev, final_idx, prev_columns, "none"
        )
        prev_stopped = fit_boosted(
            "xgboost", params_for_depth(4), X_prev, y_prev, split_groups(frame, dev)
        )
        prev_trees = int(getattr(prev_stopped, "_capscout_trees", None) or 80)
        prev_model = refit_boosted("xgboost", params_for_depth(4), prev_trees, X_prev, y_prev)
        prev_pred = predict_dollars(prev_model, X_prev_test, "xgboost")
        # Ridge on the published v3 columns. Comp and embedding columns are
        # fold-fit and are not part of this linear baseline.
        X_ridge, y_ridge, X_ridge_test = _static_ridge(frame, dev, final_idx, v3_columns)
        ridge_pred = fit_predict_ridge(X_ridge, y_ridge, X_ridge_test)
        base_pred = median_baseline(
            frame["neighbourhood_cleansed"].to_numpy(),
            frame["room_type"].to_numpy(),
            frame["bedrooms"].to_numpy(dtype=float),
            y,
            dev,
            final_idx,
            by_bedroom=False,
        )
        base_bed_pred = median_baseline(
            frame["neighbourhood_cleansed"].to_numpy(),
            frame["room_type"].to_numpy(),
            frame["bedrooms"].to_numpy(dtype=float),
            y,
            dev,
            final_idx,
            by_bedroom=True,
        )
        chosen = next(
            item for item in decision["interval_candidates"] if item["name"] == interval_method
        )
        final_metrics = _holdout_block(
            y_final,
            pred,
            final_v3[city],
            prev_pred,
            ridge_pred,
            base_pred,
            base_bed_pred,
            low,
            high,
            method_bounds,
            interval_method,
            chosen,
            city,
        )
        final_metrics["n_extra_train"] = int(held["n_extra"])
        final_metrics["n_train"] = int(len(dev))
        improvement = _improvement(
            final_metrics["baseline_room"]["mae"], final_metrics["gbm"]["mae"]
        )
        final_metrics["improvement_vs_baseline"] = improvement
        print(
            f"  {city} final R2 {final_metrics['gbm']['r2']:.3f} "
            f"MAE {final_metrics['gbm']['mae']:.2f} "
            f"coverage {final_metrics['interval']['coverage']:.3f}",
            flush=True,
        )
        _append_predictions(
            final_rows,
            city,
            frame,
            final_idx,
            y_final,
            pred,
            low,
            high,
            final_v3[city],
            base_pred,
            prev_pred,
        )

        locked_pred = locked_fit[city]["pred"]
        locked_y = locked_fit[city]["y"]
        locked_low, locked_high = _bounds(interval_method, locked_pred, city)
        locked_base = median_baseline(
            frame["neighbourhood_cleansed"].to_numpy(),
            frame["room_type"].to_numpy(),
            frame["bedrooms"].to_numpy(dtype=float),
            y,
            pack["split"]["development_idx"],
            test_idx,
            by_bedroom=False,
        )
        locked_metrics = {
            "gbm": regression_metrics(locked_y, locked_pred),
            "v3_refit": regression_metrics(locked_y, locked_v3[city]),
            "baseline_room": regression_metrics(locked_y, locked_base),
            "interval": {
                "method": interval_method,
                "nominal": INTERVAL_LEVEL,
                **interval_metrics(locked_y, locked_low, locked_high),
            },
        }
        locked_metrics["improvement_vs_baseline"] = _improvement(
            locked_metrics["baseline_room"]["mae"], locked_metrics["gbm"]["mae"]
        )
        _append_predictions(
            locked_rows,
            city,
            frame,
            test_idx,
            locked_y,
            locked_pred,
            locked_low,
            locked_high,
            locked_v3[city],
            locked_base,
            None,
        )

        confirm_pred = confirm_fit[city]["pred"]
        confirm_y = confirm_fit[city]["y"]
        confirm_low, confirm_high = _bounds(interval_method, confirm_pred, city)
        confirm_metrics = {
            "gbm": regression_metrics(confirm_y, confirm_pred),
            "v3_refit": regression_metrics(confirm_y, confirm_v3[city]),
            "interval": {
                "method": interval_method,
                "nominal": INTERVAL_LEVEL,
                **interval_metrics(confirm_y, confirm_low, confirm_high),
            },
            "n_train": int(len(pack["confirm"]["train_idx"])),
            **{
                key: pack["confirm"]["info"][key]
                for key in ("n_confirm", "removed_host", "removed_buffer", "train_after")
            },
        }
        _append_predictions(
            confirm_rows,
            city,
            frame,
            pack["confirm"]["confirm_idx"],
            confirm_y,
            confirm_pred,
            confirm_low,
            confirm_high,
            confirm_v3[city],
            None,
            None,
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
            "blend_libraries": list(spec["libraries"]),
            "blend_weights": (
                decision["stacker"]["weights"][city] if decision["stacker"].get("kept") else None
            ),
            "depth": spec["depth"],
            "monotonic": False,
            "n_trees": held["tree_counts"],
            "comp": state.get("comp"),
            "embed": state.get("embed"),
            "kernel": None,
            "text_encoder": None,
            "columns": list(held["columns"]),
            "spatial_mode": "comp" if spec["comp"] else "none",
            "text_mode": "embed" if spec["embed"] else "none",
            "pooled": False,
            "feature_set": decision["feature_set"],
            "correction": decision["correction"],
            "correction_params": decision["correction_params"][city],
            "interval_method": interval_method,
            "interval_params": decision["interval_final"][interval_method][city],
            "interval_level": INTERVAL_LEVEL,
            "param_overrides": spec.get("param_overrides") or {},
            "tail_weighted": bool(spec["tail"]),
            "uses_history": bool(spec["history"]),
            "occupancy_by_neighbourhood": {str(k): float(v) for k, v in occ_map.items()},
            "occupancy_global": float(dev_occ["occupancy_rate"].median()) if len(dev_occ) else 0.5,
            "training_latitude": frame["latitude"].to_numpy(dtype=np.float32)[dev],
            "training_longitude": frame["longitude"].to_numpy(dtype=np.float32)[dev],
            "distance_p95": float(frame.iloc[dev]["distance_to_center_km"].quantile(0.95)),
        }
        if not smoke:
            save_artifact(city, artifact)
        shap_report[city] = _blend_shap(
            held["models"], list(spec["libraries"]), held["matrix_valid"]
        )
        pdp_report[city] = _pdp_v4(held["models"], list(spec["libraries"]), held["matrix_valid"])
        city_reports[city] = {
            "display": CITIES[city]["display"],
            "snapshot": pack["stats"]["snapshot"],
            "cleaning": pack["stats"],
            "history_rows": int(len(pack["extra"])),
            "split": pack["split"]["info"],
            "confirmation_split": pack["confirm"]["info"],
            "final_split": {**pack["final"]["info"], **pack["final_score_info"]},
            "final_holdout": final_metrics,
            "locked_test": locked_metrics,
            "confirmation": confirm_metrics,
            "residuals_by_predicted_quintile": _residual_bins(y_final, pred),
            "final_holdout_by_room_type": _room_errors(frame, final_idx, pred),
            "interval_by_predicted_bin": [],
            "shap": shap_report[city],
            "partial_dependence": pdp_report[city],
            "audit": audit,
        }

    final_frame = pd.DataFrame(final_rows)
    locked_frame = pd.DataFrame(locked_rows)
    confirm_frame = pd.DataFrame(confirm_rows)
    for city in cities:
        city_reports[city]["interval_by_predicted_bin"] = _coverage_bins(
            final_frame[final_frame["city"] == city]
        )

    ablation = {}
    for city in cities:
        ablation[city] = {
            "v3": _summary([row for row in decision["base_rows"] if row["city"] == city])
        }
        for name, rows in decision["alone_rows"].items():
            ablation[city][name] = _summary([row for row in rows if row["city"] == city])
        ablation[city]["selected"] = _summary(
            [row for row in decision["rows"] if row["city"] == city]
        )

    comparison = {}
    for city in cities:
        comparison[city] = {}
        for model_name in ("baseline_room", "ridge"):
            part = [
                row for row in reference_rows if row["city"] == city and row["model"] == model_name
            ]
            comparison[city][model_name] = _summary(part)
        comparison[city]["v3"] = ablation[city]["v3"]
        comparison[city]["gbm"] = ablation[city]["selected"]

    published_v3 = _published(REPORTS / "v3_metrics.json", cities)
    published_v2 = _published(REPORTS / "v2_metrics.json", cities)
    result = {
        "study_cities": list(cities),
        "rationale": STUDY_RATIONALE,
        "protocol": _protocol_block(decision, seeds, held_columns=final_fit[cities[0]]["columns"]),
        "published_v2": published_v2,
        "published_v3": published_v3,
        "cities": city_reports,
        "development_ablation": ablation,
        "development_comparison": comparison,
        "learning_curves": curve_rows,
        "partial_dependence": pdp_report,
        "leakage_demo": (
            austin_leakage_demo(prepared["austin"]["frame"]) if "austin" in prepared else None
        ),
        "audit_passed": all(
            report["audit"]["final"]["passed"]
            and report["audit"]["split"]["passed"]
            and report["audit"]["columns"]["passed"]
            and report["audit"]["confirmation"]["passed"]
            for report in city_reports.values()
        ),
        "smoke": smoke,
        "feature_labels": {name: pretty(name) for name in final_fit[cities[0]]["columns"]},
        "holdout_note": (
            f"Seed {FINAL_SEED} is the final holdout. It was frozen before any model in "
            "this round was fit, from the cells in neither seed 7 nor seed 11. "
            f"Seed {TEST_SEED} and seed {CONFIRM_SEED} had already been read, so they are "
            "secondary comparisons and were not used to choose anything."
        ),
    }
    return result, final_frame, shap_report, locked_frame, confirm_frame


def split_groups(frame, index):
    return spatial_group_ids(
        frame["latitude"].to_numpy(dtype=float)[index],
        frame["longitude"].to_numpy(dtype=float)[index],
    )


def _static_ridge(frame, train_idx, eval_idx, columns):
    X_train, y_log, X_eval, _y_eval, _state = design_matrices(
        frame, train_idx, eval_idx, columns, "none"
    )
    return X_train, y_log, X_eval


def _improvement(baseline_mae: float, model_mae: float) -> float:
    """Share of the neighbourhood-median error the model removes. 0.20 is 20%."""
    return float((baseline_mae - model_mae) / baseline_mae)


def _holdout_block(
    y_true,
    pred,
    v3_pred,
    prev_pred,
    ridge_pred,
    base_pred,
    base_bed_pred,
    low,
    high,
    method_bounds,
    interval_method,
    chosen,
    city,
):
    return {
        "gbm": regression_metrics(y_true, pred),
        "v3_refit": regression_metrics(y_true, v3_pred),
        "previous_spec": regression_metrics(y_true, prev_pred),
        "ridge": regression_metrics(y_true, ridge_pred),
        "baseline_room": regression_metrics(y_true, base_pred),
        "baseline_room_bed": regression_metrics(y_true, base_bed_pred),
        "interval": {
            "method": interval_method,
            "nominal": INTERVAL_LEVEL,
            "development_coverage": chosen["by_city"][city]["coverage"],
            "development_worst_band": chosen["by_city"][city]["worst_band"],
            **interval_metrics(y_true, low, high),
        },
        "interval_methods": {
            name: interval_metrics(y_true, bounds[0], bounds[1])
            for name, bounds in method_bounds.items()
        },
    }


def _append_predictions(
    rows, city, frame, indices, y_true, pred, low, high, v3_pred, baseline, previous
):
    for i, row_idx in enumerate(indices):
        record = {
            "city": city,
            "listing_id": str(int(frame.at[row_idx, "id"])),
            "y_true": float(y_true[i]),
            "pred_gbm": float(pred[i]),
            "pred_lo": float(low[i]),
            "pred_hi": float(high[i]),
            "pred_v3_refit": float(v3_pred[i]),
            "room_type": str(frame.at[row_idx, "room_type"]),
            "neighbourhood": str(frame.at[row_idx, "neighbourhood_cleansed"]),
            "distance_to_center_km": float(frame.at[row_idx, "distance_to_center_km"]),
            "bedrooms": float(frame.at[row_idx, "bedrooms"]),
        }
        if baseline is not None:
            record["pred_baseline"] = float(baseline[i])
        if previous is not None:
            record["pred_previous"] = float(previous[i])
        rows.append(record)


def _published(path: Path, cities) -> dict | None:
    if not path.exists():
        return None
    archive = json.loads(path.read_text())
    published = {}
    for city in cities:
        if city not in archive.get("cities", {}):
            continue
        locked = archive["cities"][city]["locked_test"]
        published[city] = {
            "gbm": locked["gbm"],
            "interval_coverage": locked["interval"]["coverage"],
            "interval_median_width": locked["interval"]["median_width"],
            "baseline_room_mae": locked.get("baseline_room", {}).get("mae"),
        }
    return published


def _protocol_block(decision: dict, seeds, held_columns: list[str]) -> dict:
    tune = decision["tune"]
    stacker = decision["stacker"]
    # Weights are per city and small. Keep them. Drop nothing the rule replay needs.
    return {
        "test_seed": TEST_SEED,
        "confirm_seed": CONFIRM_SEED,
        "final_seed": FINAL_SEED,
        "previous_study_seed": 42,
        "inner_seeds": list(seeds),
        "v3_feature_set": "plus_census+listing_quality+listing_detail",
        "v4_base_mae": decision["base_mae"],
        "v4_trials": decision["trials"],
        "v4_kept": decision["kept"],
        "alone_development_mae": decision["alone"],
        "selected_development_mae": decision["selected_mae"],
        "libraries": decision["libraries"],
        "three_library_mae": decision["three_library_mae"],
        "tune": tune,
        "stacker_kept": stacker.get("kept"),
        "stacker_weights": stacker.get("weights"),
        "stacker_confirm_mae": stacker.get("confirm_mae"),
        "stacker_confirm_default": stacker.get("confirm_default"),
        "correction": decision["correction"],
        "correction_mae": decision["correction_mae"],
        "interval_method": decision["interval_method"],
        "interval_candidates": decision["interval_candidates"],
        "feature_set": decision["feature_set"],
        "columns": held_columns,
        "feature_set_why": FEATURE_SET_WHY,
        "selection_rule": (
            "v3's published specification is the starting point and is not re-searched. "
            "Development folds are the cells in none of seed 7, seed 11, and seed 23. "
            "Add-ons are tried in a fixed order. Each stays only if it cuts mean MAE by "
            "at least 1%. LightGBM joins the blend only on that same bar. A three-point "
            "hyperparameter grid is screened on seeds 0–1 and kept only if seeds 2–4 also "
            "clear 1%. A non-negative blend is fit on seeds 0–1 and kept only if seeds 2–4 "
            "clear 1%. Smearing and isotonic calibration are cross-fit; the simpler one "
            "wins a 1% tie, and neither is kept unless it beats the uncorrected model by 1%. "
            "The scaled log interval stays unless an asymmetric or Mondrian band improves "
            "the worst predicted-price quintile by 3 points without getting much wider. "
            "Seed 23 is scored once after that. Seed 7 and seed 11 are scored once as "
            "secondary comparisons and are not inputs."
        ),
    }


def _pdp_v4(models: dict, libraries: list[str], X: pd.DataFrame) -> dict:
    """A few curves on the rows the final model just scored. Not a tuning tool."""
    wanted = [
        name
        for name in ("bedrooms", "bathrooms", "distance_to_center_km", "census_median_income")
        if name in X.columns
    ][:4]
    sample = X.sample(n=min(400, len(X)), random_state=0)

    def _predict(matrix: pd.DataFrame) -> np.ndarray:
        stacked = [predict_dollars(models[name], matrix, name) for name in libraries]
        return np.mean(stacked, axis=0)

    curves = {}
    for name in wanted:
        if name == "bedrooms":
            grid = [0, 1, 2, 3, 4, 5]
        elif name == "bathrooms":
            grid = [1.0, 1.5, 2.0, 2.5, 3.0]
        elif name == "distance_to_center_km":
            grid = [1, 3, 5, 10, 20]
        else:
            series = sample[name].dropna()
            grid = [float(value) for value in np.quantile(series, [0.1, 0.3, 0.5, 0.7, 0.9])]
        rows = []
        for value in grid:
            edited = sample.copy()
            edited[name] = value
            rows.append({"value": float(value), "mean_pred": float(_predict(edited).mean())})
        curves[name] = rows
    return curves


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
    result, predictions, shap_report, locked, confirm = run(smoke=args.smoke)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        out = REPORTS / "smoke_metrics.json"
        out.write_text(json.dumps(_jsonable(result), indent=2))
        print(f"Wrote {out}", flush=True)
        return
    (REPORTS / "metrics.json").write_text(json.dumps(_jsonable(result), indent=2))
    predictions.to_csv(REPORTS / "final_predictions.csv", index=False)
    locked.to_csv(REPORTS / "locked_test_predictions.csv", index=False)
    confirm.to_csv(REPORTS / "confirmation_predictions.csv", index=False)
    plot_ablation(result["development_ablation"], FIGURES / "ablation_mae.png")
    plot_model_comparison(
        result["development_comparison"],
        FIGURES / "model_comparison.png",
        ["baseline_room", "ridge", "v3", "gbm"],
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
