"""What this round is allowed to choose, and where it is allowed to look.

The published v2 specification is the starting point. Development folds are
the cells in neither the seed-7 test nor the seed-11 confirmation holdout.
Nothing in this module receives the locked-test indices.

Choices, in an order written down in the protocol:

1. Add-ons, one at a time, each on top of the stack kept so far. An add-on
   stays only if mean development MAE falls by at least 1%.
2. Tree depth on that stack. Shallower wins within 1%.
3. One pooled model against the per-city blend. Pooling stays only if it
   is at least 1% better.
4. The interval. The log-residual band stays unless a challenger covers
   the expensive listings better without getting much wider or losing the
   point estimate.

The blend is CatBoost and XGBoost at equal weight, which is what v2 kept.
LightGBM and monotone constraints are not reopened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from features.columns import city_dummy
from features.text import FoldText

from evaluation.fitting import (
    design_matrices,
    fit_boosted,
    fit_quantile,
    params_for_depth,
    predict_dollars,
    predict_log,
    refit_boosted,
)
from evaluation.intervals import (
    apply_log_interval,
    apply_log_radii,
    conformal_radius,
    cqr_bounds,
    cqr_nonconformity,
    fit_log_scale,
    log_abs_residual,
    sigma_of,
)
from evaluation.metrics import interval_metrics, regression_metrics
from evaluation.protocol import (
    DEPTHS,
    INTERVAL_LEVEL,
    N_TEXT_COMPONENTS,
    SIGMA_FLOOR,
    TEXT_MAX_FEATURES,
    TEXT_MIN_DF,
    V2_BLEND,
    V2_DEPTH,
    V3_ADDON_ORDER,
    feature_label_v3,
    stack_features,
    v2_features,
)
from evaluation.select import (
    choose_depth,
    choose_interval,
    choose_pooled,
    choose_quality,
    choose_sequential,
)

UPPER_QUANTILE = 0.80
UPPER_MIN_ROWS = 20


def _mean_mae(rows: list[dict]) -> float:
    if not rows:
        raise RuntimeError("no rows to average")
    return float(np.mean([row["mae"] for row in rows]))


def _tag(rows: list[dict], tag: str) -> list[dict]:
    return [{**row, "tag": tag} for row in rows]


def _with_dummies(matrix: pd.DataFrame, city_key: str, cities: tuple[str, ...]) -> pd.DataFrame:
    out = matrix.copy()
    for key in cities:
        out[city_dummy(key)] = 1.0 if key == city_key else 0.0
    return out


def _blend_predict(models: dict, matrix: pd.DataFrame) -> np.ndarray:
    stacked = [predict_dollars(models[library], matrix, library) for library in V2_BLEND]
    return np.mean(np.vstack(stacked), axis=0)


def _fit_blend(X: pd.DataFrame, y_log: np.ndarray, groups: np.ndarray, depth: int, *, refit: bool):
    """Early-stopped blend. `refit` retrains on every row with that tree count.

    Comparison folds leave the early-stopping slice out, the same way v2
    compared candidates. The model that scores a holdout is refit so the
    slice is not thrown away.
    """
    models = {}
    tree_counts = {}
    for library in V2_BLEND:
        stopped = fit_boosted(library, params_for_depth(depth), X, y_log, groups)
        n_trees = int(getattr(stopped, "_capscout_trees", None) or 80)
        tree_counts[library] = n_trees
        if refit:
            models[library] = refit_boosted(
                library, params_for_depth(depth), n_trees, X, y_log, monotone=False
            )
        else:
            models[library] = stopped
    return models, tree_counts


def _per_city_fold(
    frame: pd.DataFrame,
    groups: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    columns: list[str],
    spatial_mode: str,
    text_mode: str,
    depth: int,
    *,
    text_encoder: FoldText | None = None,
    quantiles: bool = False,
    refit: bool = False,
):
    X_train, y_log, X_valid, y_valid, state = design_matrices(
        frame,
        train_idx,
        valid_idx,
        columns,
        spatial_mode,
        text_mode=text_mode,
        text_encoder=text_encoder,
    )
    models, tree_counts = _fit_blend(X_train, y_log, groups[train_idx], depth, refit=refit)
    pred = _blend_predict(models, X_valid)
    q_lo = q_hi = None
    q_models = None
    if quantiles:
        # Quantile models are XGBoost even though the point model is a blend.
        # A blend has no single pinball loss. The tree count is the point
        # model's, not a second search.
        n_trees = tree_counts["xgboost"]
        lo_model = fit_quantile("xgboost", params_for_depth(depth), n_trees, X_train, y_log, 0.1)
        hi_model = fit_quantile("xgboost", params_for_depth(depth), n_trees, X_train, y_log, 0.9)
        q_lo = predict_log(lo_model, X_valid, "xgboost")
        q_hi = predict_log(hi_model, X_valid, "xgboost")
        q_models = {"lo": lo_model, "hi": hi_model, "library": "xgboost"}
    return {
        "metrics": regression_metrics(y_valid, pred),
        "y": y_valid,
        "pred": pred,
        "q_lo": q_lo,
        "q_hi": q_hi,
        "state": state,
        "models": models,
        "q_models": q_models,
        "tree_counts": tree_counts,
        "columns": list(X_train.columns),
        "matrix_valid": X_valid,
    }


def _evaluate_per_city(prepared, columns, spatial_mode, text_mode, depth, *, quantiles=False):
    rows = []
    oof: dict[str, dict[int, dict]] = {}
    for city, pack in prepared.items():
        oof[city] = {}
        groups = pack["split"]["groups"]
        for seed, train_idx, valid_idx in pack["folds"]:
            fitted = _per_city_fold(
                pack["frame"],
                groups,
                train_idx,
                valid_idx,
                columns,
                spatial_mode,
                text_mode,
                depth,
                quantiles=quantiles,
            )
            rows.append({"city": city, "seed": int(seed), **fitted["metrics"]})
            oof[city][int(seed)] = {
                "y": fitted["y"],
                "pred": fitted["pred"],
                "q_lo": fitted["q_lo"],
                "q_hi": fitted["q_hi"],
            }
            print(
                f"  {city} depth {depth} seed {seed}: MAE {fitted['metrics']['mae']:.2f}",
                flush=True,
            )
    return rows, oof


def _evaluate_pooled(prepared, columns, spatial_mode, text_mode, depth, cities: tuple[str, ...]):
    """One blend for every city. Text vocabulary is shared. Kernels are not.

    A kernel uses other listings' prices, so it stays inside the city.
    SVD components have to mean the same thing in every city, so the
    vocabulary is fit on the stacked training text.
    """
    rows = []
    oof: dict[str, dict[int, dict]] = {city: {} for city in cities}
    seed_sets = [{seed for seed, _train, _valid in prepared[city]["folds"]} for city in cities]
    seeds = sorted(set.intersection(*seed_sets))
    if not seeds:
        raise RuntimeError("no selection seed is present in every city")
    for seed in seeds:
        present = {}
        for city in cities:
            present[city] = next(item for item in prepared[city]["folds"] if item[0] == seed)
        text_encoder = None
        if text_mode == "svd":
            chunks = [
                prepared[city]["frame"]["listing_text"].to_numpy()[present[city][1]]
                for city in cities
            ]
            text_encoder = FoldText(
                n_components=N_TEXT_COMPONENTS,
                max_features=TEXT_MAX_FEATURES,
                min_df=TEXT_MIN_DF,
            ).fit(np.concatenate(chunks))
        train_parts = []
        y_parts = []
        group_parts = []
        valid = {}
        for city in cities:
            _seed, train_idx, valid_idx = present[city]
            fitted_state_only = design_matrices(
                prepared[city]["frame"],
                train_idx,
                valid_idx,
                columns,
                spatial_mode,
                text_mode=text_mode,
                text_encoder=text_encoder,
            )
            X_train, y_log, X_valid, y_valid, _state = fitted_state_only
            X_train = _with_dummies(X_train, city, cities)
            X_valid = _with_dummies(X_valid, city, cities)
            train_parts.append(X_train)
            y_parts.append(y_log)
            groups = prepared[city]["split"]["groups"][train_idx]
            group_parts.append(np.array([f"{city}|{group}" for group in groups]))
            valid[city] = (X_valid, y_valid)
        column_lists = [list(part.columns) for part in train_parts]
        if any(columns != column_lists[0] for columns in column_lists[1:]):
            raise RuntimeError("Pooled cities produced different columns")
        X_train = pd.concat(train_parts, ignore_index=True)
        y_log = np.concatenate(y_parts)
        groups = np.concatenate(group_parts)
        models, _trees = _fit_blend(X_train, y_log, groups, depth, refit=False)
        for city, (X_valid, y_valid) in valid.items():
            pred = _blend_predict(models, X_valid)
            metrics = regression_metrics(y_valid, pred)
            rows.append({"city": city, "seed": int(seed), **metrics})
            oof[city][int(seed)] = {"y": y_valid, "pred": pred, "q_lo": None, "q_hi": None}
            print(f"  pooled {city} seed {seed}: MAE {metrics['mae']:.2f}", flush=True)
    return rows, oof


def _context_from_folds(method: str, folds: list[dict]) -> dict:
    y = np.concatenate([fold["y"] for fold in folds])
    pred = np.concatenate([fold["pred"] for fold in folds])
    residual = log_abs_residual(y, pred)
    if method == "log_conformal":
        return {"radius": conformal_radius(residual)}
    if method == "scaled_log":
        intercept, slope = fit_log_scale(pred, residual)
        sigma = sigma_of(pred, intercept, slope, SIGMA_FLOOR)
        return {
            "a": intercept,
            "b": slope,
            "floor": SIGMA_FLOOR,
            "quantile": conformal_radius(residual / sigma),
        }
    if method == "cqr":
        scores = np.concatenate(
            [cqr_nonconformity(fold["y"], fold["q_lo"], fold["q_hi"]) for fold in folds]
        )
        return {"qhat": conformal_radius(scores)}
    raise ValueError(method)


def _bounds(method: str, fold: dict, context: dict) -> tuple[np.ndarray, np.ndarray]:
    if method == "log_conformal":
        return apply_log_interval(fold["pred"], context["radius"])
    if method == "scaled_log":
        sigma = sigma_of(fold["pred"], context["a"], context["b"], context["floor"])
        return apply_log_radii(fold["pred"], context["quantile"] * sigma)
    low, high = cqr_bounds(fold["q_lo"], fold["q_hi"], context["qhat"])
    return low, high


def _fold_interval_stats(y, pred, low, high) -> dict:
    covered = (y >= low) & (y <= high)
    inside = (pred >= low) & (pred <= high)
    edge = float(np.quantile(pred, UPPER_QUANTILE))
    tail = pred >= edge
    if int(tail.sum()) >= UPPER_MIN_ROWS:
        upper = float(covered[tail].mean())
    else:
        upper = float("nan")
    width = interval_metrics(y, low, high)
    return {
        "coverage": float(covered.mean()),
        "upper": upper,
        "median_width": width["median_width"],
        "point_inside": float(inside.mean()),
    }


def _nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    return float(array.mean())


def _calibrate_city(method: str, by_seed: dict[int, dict]) -> dict:
    folds = [by_seed[seed] for seed in sorted(by_seed)]
    stats = []
    for index, fold in enumerate(folds):
        others = folds if len(folds) == 1 else [item for i, item in enumerate(folds) if i != index]
        context = _context_from_folds(method, others)
        low, high = _bounds(method, fold, context)
        stats.append(_fold_interval_stats(fold["y"], fold["pred"], low, high))
    return {
        "coverage": float(np.mean([item["coverage"] for item in stats])),
        "upper": _nanmean([item["upper"] for item in stats]),
        "median_width": float(np.mean([item["median_width"] for item in stats])),
        "point_inside": float(np.mean([item["point_inside"] for item in stats])),
        "final": _context_from_folds(method, folds),
    }


def _city_mean(per_city: dict[str, dict], key: str) -> float:
    return float(np.mean([per_city[city][key] for city in per_city]))


def _city_min(per_city: dict[str, dict], key: str) -> float:
    return float(np.min([per_city[city][key] for city in per_city]))


def calibrate_intervals(oof: dict[str, dict[int, dict]]) -> dict:
    """Cross-fitted coverage on the selection folds. Does not see a holdout."""
    candidates = []
    final: dict[str, dict] = {}
    per_method_cities: dict[str, dict] = {}
    for method in ("log_conformal", "scaled_log", "cqr"):
        per_city = {city: _calibrate_city(method, oof[city]) for city in oof}
        per_method_cities[method] = per_city
        summary = {
            "name": method,
            "mean_coverage": _city_mean(per_city, "coverage"),
            "worst_coverage": _city_min(per_city, "coverage"),
            "worst_upper_coverage": _city_min(per_city, "upper"),
            "mean_median_width": _city_mean(per_city, "median_width"),
            "point_inside": _city_min(per_city, "point_inside"),
            "by_city": {
                city: {
                    "coverage": per_city[city]["coverage"],
                    "upper_coverage": per_city[city]["upper"],
                    "median_width": per_city[city]["median_width"],
                    "point_inside": per_city[city]["point_inside"],
                }
                for city in per_city
            },
        }
        candidates.append(summary)
        final[method] = {city: per_city[city]["final"] for city in per_city}
    method = choose_interval(candidates)
    return {"method": method, "candidates": candidates, "final": final}


def _attach_quantiles(prepared, oof, columns, spatial_mode, text_mode, depth):
    """Per-city quantile predictions on the same selection folds.

    The point prediction in `oof` is left as it is, so a pooled point model
    keeps its predictions. The quantile band is always fit inside the city.
    """
    for city, pack in prepared.items():
        groups = pack["split"]["groups"]
        for seed, train_idx, valid_idx in pack["folds"]:
            fitted = _per_city_fold(
                pack["frame"],
                groups,
                train_idx,
                valid_idx,
                columns,
                spatial_mode,
                text_mode,
                depth,
                quantiles=True,
            )
            slot = oof[city][int(seed)]
            if len(slot["y"]) != len(fitted["y"]):
                raise RuntimeError(f"{city} seed {seed} quantile rows do not match the point model")
            slot["q_lo"] = fitted["q_lo"]
            slot["q_hi"] = fitted["q_hi"]


def select_v3(prepared: dict, *, smoke: bool = False) -> dict:
    """Development-only selection. `prepared[city]['folds']` is the only sample."""
    base_columns, _base_spatial = v2_features()
    bucket: list[dict] = []
    print("v2 specification, on the selection folds", flush=True)
    base_rows, _base_oof = _evaluate_per_city(
        prepared, base_columns, "none", "none", V2_DEPTH, quantiles=False
    )
    bucket.extend(_tag(base_rows, "v3:v2"))
    base_mae = _mean_mae(base_rows)
    print(f"  v2 MAE {base_mae:.2f}", flush=True)

    print("Add-ons alone, for the ablation. Not used to reorder the stack.", flush=True)
    alone = {}
    for name in V3_ADDON_ORDER:
        columns, spatial, text = stack_features(base_columns, [name])
        rows, _oof = _evaluate_per_city(prepared, columns, spatial, text, V2_DEPTH)
        bucket.extend(_tag(rows, f"v3:alone:{name}"))
        alone[name] = _mean_mae(rows)
        print(f"  alone {name} MAE {alone[name]:.2f}", flush=True)

    print("Sequential add-ons", flush=True)
    kept: list[str] = []
    trials = []
    current = base_mae
    for name in V3_ADDON_ORDER:
        trial = kept + [name]
        columns, spatial, text = stack_features(base_columns, trial)
        rows, _oof = _evaluate_per_city(prepared, columns, spatial, text, V2_DEPTH)
        mae = _mean_mae(rows)
        bucket.extend(_tag(rows, f"v3:trial:{name}"))
        trials.append({"name": name, "mae": mae})
        if choose_quality(current, mae):
            kept = trial
            current = mae
            print(f"  kept {name} MAE {mae:.2f}", flush=True)
        else:
            print(f"  dropped {name} MAE {mae:.2f}", flush=True)
    replay = choose_sequential(base_mae, [(item["name"], item["mae"]) for item in trials])
    if replay != kept:
        raise RuntimeError(f"Sequential replay disagreed: {replay} vs {kept}")

    columns, spatial, text = stack_features(base_columns, kept)
    print("Depth on the kept stack", flush=True)
    depth_rows = {}
    depth_mae = {}
    for depth in ((V2_DEPTH,) if smoke else DEPTHS):
        rows, _oof = _evaluate_per_city(prepared, columns, spatial, text, depth)
        bucket.extend(_tag(rows, f"v3:depth:{depth}"))
        depth_rows[depth] = rows
        depth_mae[depth] = _mean_mae(rows)
        print(f"  depth {depth} MAE {depth_mae[depth]:.2f}", flush=True)
    chosen_depth = choose_depth([(depth, mae) for depth, mae in depth_mae.items()])
    print(f"Chosen depth {chosen_depth}", flush=True)

    cities = tuple(prepared)
    pooled = False
    pooled_mae = None
    per_city_mae = depth_mae[chosen_depth]
    pooled_oof = None
    if len(cities) >= 3 and not smoke:
        print("Pooled model", flush=True)
        pooled_rows, pooled_oof = _evaluate_pooled(
            prepared, columns, spatial, text, chosen_depth, cities
        )
        bucket.extend(_tag(pooled_rows, "v3:pooled"))
        pooled_mae = _mean_mae(pooled_rows)
        pooled = choose_pooled(per_city_mae, pooled_mae)
        print(f"  pooled MAE {pooled_mae:.2f} kept {pooled}", flush=True)

    print("Quantile fits for the interval comparison", flush=True)
    if pooled:
        point_oof = pooled_oof
        selected_rows = [row for row in bucket if row["tag"] == "v3:pooled"]
    else:
        point_oof = None
        selected_rows = depth_rows[chosen_depth]
    # Quantiles are fit per city either way. Rebuild the point OOF for the
    # per-city model so the log-residual band uses the same predictions as
    # the quantile scores when pooling was not chosen.
    quantile_rows, quantile_oof = _evaluate_per_city(
        prepared, columns, spatial, text, chosen_depth, quantiles=True
    )
    if pooled:
        _attach_quantiles(prepared, point_oof, columns, spatial, text, chosen_depth)
    else:
        point_oof = quantile_oof
        selected_rows = quantile_rows
    bucket.extend(_tag(selected_rows, "v3:selected"))

    print("Interval methods, on the selection folds", flush=True)
    interval = calibrate_intervals(point_oof)
    print(f"  interval {interval['method']}", flush=True)
    for candidate in interval["candidates"]:
        print(
            f"    {candidate['name']}: worst coverage {candidate['worst_coverage']:.3f} "
            f"upper {candidate['worst_upper_coverage']:.3f} "
            f"width ${candidate['mean_median_width']:.0f}",
            flush=True,
        )

    return {
        "columns": columns,
        "spatial_mode": spatial,
        "text_mode": text,
        "kept": kept,
        "pooled": pooled,
        "depth": int(chosen_depth),
        "base_mae": base_mae,
        "trials": trials,
        "alone": alone,
        "depth_mae": {str(depth): mae for depth, mae in depth_mae.items()},
        "per_city_mae": per_city_mae,
        "pooled_mae": pooled_mae,
        "rows": bucket,
        "interval_method": interval["method"],
        "interval_candidates": interval["candidates"],
        "interval_final": interval["final"],
        "feature_set": feature_label_v3(kept, pooled),
        "blend": list(V2_BLEND),
    }


def fit_holdout(prepared: dict, train_idxs: dict, eval_idxs: dict, spec: dict, *, refit: bool):
    """Fit the chosen point model and per-city quantile models. Score `eval_idxs`.

    `refit` is true for a holdout score: early stopping picks the tree count,
    then the model is retrained on every training row. Comparison folds pass
    false and are not used here.
    """
    cities = tuple(prepared)
    columns = spec["columns"]
    spatial = spec["spatial_mode"]
    text = spec["text_mode"]
    depth = spec["depth"]
    out = {}
    if spec["pooled"]:
        text_encoder = None
        if text == "svd":
            chunks = [
                prepared[city]["frame"]["listing_text"].to_numpy()[train_idxs[city]]
                for city in cities
            ]
            text_encoder = FoldText(
                n_components=N_TEXT_COMPONENTS,
                max_features=TEXT_MAX_FEATURES,
                min_df=TEXT_MIN_DF,
            ).fit(np.concatenate(chunks))
        train_parts = []
        y_parts = []
        group_parts = []
        states = {}
        valid = {}
        for city in cities:
            X_train, y_log, X_valid, _y_valid, state = design_matrices(
                prepared[city]["frame"],
                train_idxs[city],
                eval_idxs[city],
                columns,
                spatial,
                text_mode=text,
                text_encoder=text_encoder,
            )
            states[city] = state
            train_parts.append(_with_dummies(X_train, city, cities))
            y_parts.append(y_log)
            groups = prepared[city]["split"]["groups"][train_idxs[city]]
            group_parts.append(np.array([f"{city}|{group}" for group in groups]))
            valid[city] = _with_dummies(X_valid, city, cities)
        X_train = pd.concat(train_parts, ignore_index=True)
        y_log = np.concatenate(y_parts)
        groups = np.concatenate(group_parts)
        models, tree_counts = _fit_blend(X_train, y_log, groups, depth, refit=refit)
        shared_columns = list(X_train.columns)
        for city in cities:
            pred = _blend_predict(models, valid[city])
            out[city] = {
                "pred": pred,
                "models": models,
                "state": states[city],
                "tree_counts": tree_counts,
                "columns": shared_columns,
                "pooled_text": text_encoder,
                "matrix_valid": valid[city],
            }
    else:
        for city in cities:
            fitted = _per_city_fold(
                prepared[city]["frame"],
                prepared[city]["split"]["groups"],
                train_idxs[city],
                eval_idxs[city],
                columns,
                spatial,
                text,
                depth,
                quantiles=False,
                refit=refit,
            )
            out[city] = fitted

    for city in cities:
        quant = _per_city_fold(
            prepared[city]["frame"],
            prepared[city]["split"]["groups"],
            train_idxs[city],
            eval_idxs[city],
            columns,
            spatial,
            text,
            depth,
            quantiles=True,
            refit=refit,
        )
        out[city]["q_lo"] = quant["q_lo"]
        out[city]["q_hi"] = quant["q_hi"]
        out[city]["q_models"] = quant["q_models"]
        y = prepared[city]["frame"]["price_clean"].to_numpy(dtype=float)[eval_idxs[city]]
        out[city]["y"] = y
        if len(y) != len(out[city]["pred"]):
            raise RuntimeError(f"{city} prediction length does not match the holdout")
    return out


def apply_interval(method: str, pred: np.ndarray, q_lo, q_hi, params: dict):
    """Dollar bounds from the calibration chosen on the selection folds."""
    fold = {"pred": pred, "q_lo": q_lo, "q_hi": q_hi}
    low, high = _bounds(method, fold, params)
    return low, high


def interval_nominal() -> float:
    return INTERVAL_LEVEL
