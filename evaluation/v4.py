"""What this round is allowed to choose, and where it is allowed to look.

The published v3 specification is the starting point. It is not re-searched.
Development folds are the cells in none of the three holdouts: seed 7, seed
11, and the seed-23 final holdout. Nothing in this module receives the
final-holdout labels while it is still choosing.

Choices, in the order written down in the protocol:

1. Add-ons, one at a time, each on top of the stack kept so far. An add-on
   stays only if mean development MAE falls by at least 1%.
2. LightGBM joins the equal-weight blend only if that trio is at least 1%
   better than CatBoost and XGBoost.
3. A three-point hyperparameter grid, screened on seeds 0–1 and kept only
   if seeds 2–4 also clear 1%.
4. A non-negative blend of the chosen libraries. The weights are fit on
   seeds 0–1 and kept only if seeds 2–4 clear 1%.
5. Duan smearing, then isotonic calibration. Cross-fit. The simpler one
   wins a 1% tie. Neither is kept unless it beats the uncorrected MAE by 1%.
6. The interval. The scaled log band stays unless an asymmetric or a
   Mondrian asymmetric band covers the worst predicted-price quintile
   better, without getting much wider.

The final holdout is scored once, after this function returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from features.comps import MatchedComps
from features.embed import FoldEmbed
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression

from evaluation.blocking import protect_train
from evaluation.fitting import (
    fit_boosted,
    params_for_depth,
    predict_dollars,
    refit_boosted,
)
from evaluation.intervals import (
    INTERVAL_LEVEL,
    apply_log_radii,
    apply_mondrian,
    apply_signed_interval,
    assign_band,
    asymmetric_quantiles,
    band_edges,
    conformal_radius,
    fit_log_scale,
    fit_mondrian,
    log_abs_residual,
    sigma_of,
    signed_log_residual,
)
from evaluation.metrics import interval_metrics, regression_metrics
from evaluation.protocol import (
    BAND_MIN_EVAL,
    MONDRIAN_BANDS,
    MONDRIAN_MIN_ROWS,
    SIGMA_FLOOR,
    TAIL_WEIGHT_HIGH,
    TUNE_CONFIRM_SEEDS,
    TUNE_GRID,
    TUNE_SCREEN_SEEDS,
    V2_BLEND,
    V2_DEPTH,
    V4_ADDON_ORDER,
    v3_features,
)
from evaluation.select import (
    choose_band_interval,
    choose_confirmed,
    choose_correction,
    choose_quality,
    choose_sequential,
)
from evaluation.splits import spatial_group_ids

# Re-exported so a caller does not import the interval module for the level.
NOMINAL_COVERAGE = INTERVAL_LEVEL


def tail_weights(prices: np.ndarray) -> np.ndarray:
    """1 at the median log price, 3 at the 90th percentile, clipped to [1, 3]."""
    log_price = np.log1p(np.asarray(prices, dtype=float))
    midpoint = float(np.quantile(log_price, 0.50))
    high = float(np.quantile(log_price, 0.90))
    span = max(high - midpoint, 1e-6)
    raw = 1.0 + (TAIL_WEIGHT_HIGH - 1.0) * (log_price - midpoint) / span
    return np.clip(raw, 1.0, TAIL_WEIGHT_HIGH)


def _mean_mae(rows: list[dict]) -> float:
    if not rows:
        raise RuntimeError("no rows to average")
    return float(np.mean([row["mae"] for row in rows]))


def _empty_extra() -> pd.DataFrame:
    return pd.DataFrame()


def filter_extra(
    extra: pd.DataFrame | None,
    frame: pd.DataFrame,
    eval_idx: np.ndarray,
    blocked_idx: np.ndarray,
) -> pd.DataFrame:
    """Drop historical rows that sit in a held-out cell, share a host, or are within 1 km.

    `blocked_idx` is the final holdout during selection, so a development
    fold cannot train on the past prices of the cells it is not allowed to
    see. At score time the holdout itself is `eval_idx` and `blocked_idx`
    is empty.
    """
    if extra is None or len(extra) == 0:
        return _empty_extra()
    block = np.unique(
        np.concatenate([np.asarray(eval_idx, dtype=int), np.asarray(blocked_idx, dtype=int)])
    )
    if len(block) == 0:
        return extra.reset_index(drop=True)
    blocked_groups = set(
        spatial_group_ids(
            frame["latitude"].to_numpy(dtype=float)[block],
            frame["longitude"].to_numpy(dtype=float)[block],
        )
    )
    extra_groups = spatial_group_ids(
        extra["latitude"].to_numpy(dtype=float),
        extra["longitude"].to_numpy(dtype=float),
    )
    kept = extra.loc[~np.isin(extra_groups, list(blocked_groups))].reset_index(drop=True)
    if len(kept) == 0:
        return kept
    lat = np.concatenate(
        [kept["latitude"].to_numpy(dtype=float), frame["latitude"].to_numpy(dtype=float)[block]]
    )
    lon = np.concatenate(
        [
            kept["longitude"].to_numpy(dtype=float),
            frame["longitude"].to_numpy(dtype=float)[block],
        ]
    )
    hosts = np.concatenate([kept["host_id"].to_numpy(), frame["host_id"].to_numpy()[block]])
    train_idx = np.arange(len(kept))
    test_idx = np.arange(len(kept), len(kept) + len(block))
    surviving, _info = protect_train(lat, lon, hosts, train_idx, test_idx)
    return kept.iloc[surviving].reset_index(drop=True)


def _base_spec() -> dict:
    columns, _spatial, _text = v3_features()
    return {
        "base_columns": list(columns),
        "comp": False,
        "embed": False,
        "history": False,
        "tail": False,
        "libraries": list(V2_BLEND),
        "param_overrides": {},
        "depth": V2_DEPTH,
    }


def spec_from_kept(kept: list[str], **overrides) -> dict:
    spec = _base_spec()
    spec["comp"] = "matched_comps" in kept
    spec["embed"] = "text_embed" in kept
    spec["history"] = "history" in kept
    spec["tail"] = "tail_weight" in kept
    spec.update(overrides)
    return spec


def build_matrices(
    frame: pd.DataFrame,
    extra: pd.DataFrame | None,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    blocked_idx: np.ndarray,
    spec: dict,
):
    """Training matrix, including historical rows when the spec says so.

    Comparables and the embedding PCA are fit on that training matrix only.
    The evaluated rows are always from the study snapshot.
    """
    train_rows = frame.iloc[np.asarray(train_idx, dtype=int)].copy()
    train_rows["months_before"] = train_rows.get("months_before", 0.0)
    if "months_before" not in frame.columns:
        train_rows["months_before"] = 0.0
    parts = [train_rows]
    n_extra = 0
    if spec["history"]:
        historic = filter_extra(extra, frame, eval_idx, blocked_idx)
        if len(historic):
            parts.append(historic)
            n_extra = int(len(historic))
    train = pd.concat(parts, ignore_index=True)
    eval_rows = frame.iloc[np.asarray(eval_idx, dtype=int)].copy()
    eval_rows["months_before"] = 0.0
    columns = list(spec["base_columns"])
    if spec["history"]:
        columns = columns + ["months_before"]
    X_train = train[columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    X_eval = eval_rows[columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y_log = np.log1p(train["price_clean"].to_numpy(dtype=float))
    y_eval = eval_rows["price_clean"].to_numpy(dtype=float)
    state: dict = {"n_extra": n_extra}
    if spec["comp"]:
        comps = MatchedComps().fit(
            train["latitude"].to_numpy(dtype=float),
            train["longitude"].to_numpy(dtype=float),
            y_log,
            train["room_type"].to_numpy(),
            train["bedrooms"].to_numpy(dtype=float),
        )
        X_train["comp_log_price"] = comps.transform(
            train["latitude"].to_numpy(dtype=float),
            train["longitude"].to_numpy(dtype=float),
            train["room_type"].to_numpy(),
            train["bedrooms"].to_numpy(dtype=float),
            row_index=np.arange(len(train)),
        )
        X_eval["comp_log_price"] = comps.transform(
            eval_rows["latitude"].to_numpy(dtype=float),
            eval_rows["longitude"].to_numpy(dtype=float),
            eval_rows["room_type"].to_numpy(),
            eval_rows["bedrooms"].to_numpy(dtype=float),
            row_index=None,
        )
        state["comp"] = comps
    if spec["embed"]:
        embed = FoldEmbed().fit(train["listing_text"].to_numpy())
        train_embed = embed.transform(train["listing_text"].to_numpy())
        eval_embed = embed.transform(eval_rows["listing_text"].to_numpy())
        for index, name in enumerate(embed.column_names):
            X_train[name] = train_embed[:, index]
            X_eval[name] = eval_embed[:, index]
        state["embed"] = embed
    weights = tail_weights(train["price_clean"].to_numpy(dtype=float)) if spec["tail"] else None
    groups = spatial_group_ids(
        train["latitude"].to_numpy(dtype=float),
        train["longitude"].to_numpy(dtype=float),
    )
    return X_train, y_log, X_eval, y_eval, weights, groups, state


def _blend(by_library: dict[str, np.ndarray], libraries: list[str], weights: dict | None):
    if weights:
        total = np.zeros(len(by_library[libraries[0]]), dtype=float)
        for library in libraries:
            total = total + float(weights[library]) * by_library[library]
        return total
    stacked = np.vstack([by_library[library] for library in libraries])
    return stacked.mean(axis=0)


def _fit_libraries(X_train, y_log, groups, weights, X_eval, spec, *, refit: bool):
    params = params_for_depth(spec["depth"])
    params.update(spec.get("param_overrides") or {})
    by_library = {}
    models = {}
    tree_counts = {}
    for library in spec["libraries"]:
        stopped = fit_boosted(library, params, X_train, y_log, groups, sample_weight=weights)
        n_trees = int(getattr(stopped, "_capscout_trees", None) or 80)
        tree_counts[library] = n_trees
        if refit:
            model = refit_boosted(library, params, n_trees, X_train, y_log, sample_weight=weights)
        else:
            model = stopped
        models[library] = model
        by_library[library] = predict_dollars(model, X_eval, library)
    return by_library, models, tree_counts


def evaluate(prepared: dict, spec: dict, seeds: tuple[int, ...] | None = None):
    """Development MAE. `seeds` limits the folds; None uses every prepared fold."""
    rows = []
    oof: dict[str, dict[int, dict]] = {}
    for city, pack in prepared.items():
        oof[city] = {}
        blocked = pack["blocked_idx"]
        for seed, train_idx, valid_idx in pack["folds"]:
            if seeds is not None and int(seed) not in seeds:
                continue
            X_train, y_log, X_valid, y_valid, weights, groups, _state = build_matrices(
                pack["frame"], pack["extra"], train_idx, valid_idx, blocked, spec
            )
            by_library, _models, _trees = _fit_libraries(
                X_train, y_log, groups, weights, X_valid, spec, refit=False
            )
            pred = _blend(by_library, spec["libraries"], None)
            metrics = regression_metrics(y_valid, pred)
            rows.append({"city": city, "seed": int(seed), **metrics})
            oof[city][int(seed)] = {"y": y_valid, "pred": pred, "by_library": by_library}
            print(
                f"    {city} seed {seed}: MAE {metrics['mae']:.2f}",
                flush=True,
            )
    return rows, oof


def _has_seeds(prepared: dict, seeds: tuple[int, ...]) -> bool:
    wanted = set(int(seed) for seed in seeds)
    for pack in prepared.values():
        have = {int(seed) for seed, _train, _valid in pack["folds"]}
        if not wanted.issubset(have):
            return False
    return True


def _mae_on(oof: dict, seeds: tuple[int, ...]) -> float:
    scores = []
    for by_seed in oof.values():
        for seed in seeds:
            slot = by_seed[int(seed)]
            scores.append(regression_metrics(slot["y"], slot["pred"])["mae"])
    return float(np.mean(scores))


def _reweight(oof: dict, libraries: list[str], weights_by_city: dict) -> dict:
    """Equal-library predictions replaced by the city weights. Libraries stay."""
    out = {}
    for city, by_seed in oof.items():
        out[city] = {}
        for seed, slot in by_seed.items():
            pred = _blend(slot["by_library"], libraries, weights_by_city[city])
            out[city][seed] = {"y": slot["y"], "pred": pred, "by_library": slot["by_library"]}
    return out


def _stack_weights(oof: dict, libraries: list[str], seeds: tuple[int, ...]) -> dict:
    weights = {}
    for city, by_seed in oof.items():
        parts = []
        targets = []
        for seed in seeds:
            slot = by_seed[int(seed)]
            parts.append(np.column_stack([slot["by_library"][library] for library in libraries]))
            targets.append(slot["y"])
        design = np.vstack(parts)
        y = np.concatenate(targets)
        raw, _residual = nnls(design, y)
        if float(raw.sum()) <= 1e-8:
            raw = np.ones(len(libraries), dtype=float)
        raw = raw / raw.sum()
        weights[city] = {
            library: float(weight) for library, weight in zip(libraries, raw, strict=True)
        }
    return weights


def smear_factor(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Duan's factor: mean exp(log residual). Turns a log prediction into a mean."""
    residual = signed_log_residual(y_true, pred)
    return float(np.mean(np.exp(residual)))


def apply_smear(pred: np.ndarray, factor: float) -> np.ndarray:
    pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
    return np.maximum(np.exp(np.log1p(pred)) * float(factor) - 1.0, 0.0)


class IsotonicMap:
    """Monotone map from predicted dollars to actual dollars. Fit on out-of-fold rows."""

    def __init__(self) -> None:
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, pred: np.ndarray, y_true: np.ndarray) -> IsotonicMap:
        pred = np.asarray(pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        order = np.argsort(pred, kind="mergesort")
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(pred[order], y_true[order])
        self.x_ = np.asarray(model.X_thresholds_, dtype=float)
        self.y_ = np.asarray(model.y_thresholds_, dtype=float)
        return self

    def apply(self, pred: np.ndarray) -> np.ndarray:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("IsotonicMap.fit must be called before apply")
        return np.interp(np.asarray(pred, dtype=float), self.x_, self.y_)


def _concat_slots(slots: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([slot["y"] for slot in slots]),
        np.concatenate([slot["pred"] for slot in slots]),
    )


def cross_correct(oof: dict, method: str) -> tuple[list[dict], dict]:
    """Leave-one-seed-out correction. One seed calibrates on itself, which is smoke only."""
    rows = []
    corrected: dict[str, dict[int, dict]] = {}
    for city, by_seed in oof.items():
        corrected[city] = {}
        seeds = sorted(by_seed)
        for index, seed in enumerate(seeds):
            others = (
                [by_seed[item] for item in seeds]
                if len(seeds) == 1
                else [by_seed[item] for i, item in enumerate(seeds) if i != index]
            )
            y_cal, pred_cal = _concat_slots(others)
            pred = by_seed[seed]["pred"]
            y_true = by_seed[seed]["y"]
            if method == "smear":
                adjusted = apply_smear(pred, smear_factor(y_cal, pred_cal))
            elif method == "isotonic":
                adjusted = IsotonicMap().fit(pred_cal, y_cal).apply(pred)
            else:
                raise ValueError(method)
            corrected[city][seed] = {
                "y": y_true,
                "pred": adjusted,
                "by_library": by_seed[seed]["by_library"],
            }
            rows.append({"city": city, "seed": int(seed), **regression_metrics(y_true, adjusted)})
    return rows, corrected


def fit_correction_params(oof: dict, method: str) -> dict:
    """The map applied to a holdout. Fit on every development fold, not on the holdout."""
    params = {}
    for city, by_seed in oof.items():
        y_true, pred = _concat_slots([by_seed[seed] for seed in sorted(by_seed)])
        if method == "smear":
            params[city] = {"factor": smear_factor(y_true, pred)}
        elif method == "isotonic":
            model = IsotonicMap().fit(pred, y_true)
            params[city] = {"x": model.x_.tolist(), "y": model.y_.tolist()}
        elif method == "none":
            params[city] = {}
        else:
            raise ValueError(method)
    return params


def apply_correction(pred: np.ndarray, method: str, params: dict) -> np.ndarray:
    if method == "none" or not params:
        return np.asarray(pred, dtype=float)
    if method == "smear":
        return apply_smear(pred, float(params["factor"]))
    if method == "isotonic":
        return np.interp(
            np.asarray(pred, dtype=float),
            np.asarray(params["x"], dtype=float),
            np.asarray(params["y"], dtype=float),
        )
    raise ValueError(method)


def _fold_stats(y_true, pred, low, high) -> dict:
    covered = (y_true >= low) & (y_true <= high)
    inside = (pred >= low) & (pred <= high)
    width = interval_metrics(y_true, low, high)
    edges = band_edges(pred, MONDRIAN_BANDS)
    bands = assign_band(pred, edges)
    band_coverages = []
    for band in range(MONDRIAN_BANDS):
        mask = bands == band
        if int(mask.sum()) < BAND_MIN_EVAL:
            continue
        band_coverages.append(float(covered[mask].mean()))
    worst_band = float(min(band_coverages)) if band_coverages else float("nan")
    upper_edge = float(np.quantile(pred, 0.80))
    tail = pred >= upper_edge
    upper = float(covered[tail].mean()) if int(tail.sum()) >= 20 else float("nan")
    return {
        "coverage": float(covered.mean()),
        "upper": upper,
        "worst_band": worst_band,
        "median_width": width["median_width"],
        "point_inside": float(inside.mean()),
    }


def _interval_context(method: str, slots: list[dict]) -> dict:
    y_true, pred = _concat_slots(slots)
    if method == "scaled_log":
        residual = log_abs_residual(y_true, pred)
        intercept, slope = fit_log_scale(pred, residual)
        sigma = sigma_of(pred, intercept, slope, SIGMA_FLOOR)
        return {
            "a": intercept,
            "b": slope,
            "floor": SIGMA_FLOOR,
            "quantile": conformal_radius(residual / sigma),
        }
    if method == "asymmetric":
        low, high = asymmetric_quantiles(signed_log_residual(y_true, pred))
        return {"q_lo": low, "q_hi": high}
    if method == "mondrian_asymmetric":
        return fit_mondrian(
            pred,
            signed_log_residual(y_true, pred),
            n_bands=MONDRIAN_BANDS,
            min_rows=MONDRIAN_MIN_ROWS,
        )
    raise ValueError(method)


def _interval_bounds(method: str, pred: np.ndarray, context: dict):
    if method == "scaled_log":
        sigma = sigma_of(pred, context["a"], context["b"], context["floor"])
        return apply_log_radii(pred, context["quantile"] * sigma)
    if method == "asymmetric":
        return apply_signed_interval(pred, context["q_lo"], context["q_hi"])
    if method == "mondrian_asymmetric":
        return apply_mondrian(pred, context)
    raise ValueError(method)


def _nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    return float(array.mean())


def _calibrate_method(method: str, oof: dict) -> tuple[dict, dict]:
    per_city = {}
    final = {}
    for city, by_seed in oof.items():
        seeds = sorted(by_seed)
        stats = []
        for index, seed in enumerate(seeds):
            others = (
                [by_seed[item] for item in seeds]
                if len(seeds) == 1
                else [by_seed[item] for i, item in enumerate(seeds) if i != index]
            )
            context = _interval_context(method, others)
            low, high = _interval_bounds(method, by_seed[seed]["pred"], context)
            stats.append(_fold_stats(by_seed[seed]["y"], by_seed[seed]["pred"], low, high))
        per_city[city] = {
            "coverage": float(np.mean([item["coverage"] for item in stats])),
            "upper": _nanmean([item["upper"] for item in stats]),
            "worst_band": _nanmean([item["worst_band"] for item in stats]),
            "median_width": float(np.mean([item["median_width"] for item in stats])),
            "point_inside": float(np.mean([item["point_inside"] for item in stats])),
        }
        final[city] = _interval_context(method, [by_seed[seed] for seed in seeds])
    summary = {
        "name": method,
        "mean_coverage": float(np.mean([item["coverage"] for item in per_city.values()])),
        "worst_coverage": float(np.min([item["coverage"] for item in per_city.values()])),
        "worst_upper_coverage": float(np.min([item["upper"] for item in per_city.values()])),
        "worst_band_coverage": float(np.min([item["worst_band"] for item in per_city.values()])),
        "mean_median_width": float(np.mean([item["median_width"] for item in per_city.values()])),
        "point_inside": float(np.min([item["point_inside"] for item in per_city.values()])),
        "by_city": per_city,
    }
    return summary, final


def calibrate_intervals(oof: dict) -> dict:
    """Scaled log band, unless a pre-registered challenger covers the worst band better."""
    candidates = []
    final = {}
    for method in ("scaled_log", "asymmetric", "mondrian_asymmetric"):
        summary, params = _calibrate_method(method, oof)
        candidates.append(summary)
        final[method] = params
        print(
            f"    {method}: worst city {summary['worst_coverage']:.3f} "
            f"worst band {summary['worst_band_coverage']:.3f} "
            f"width ${summary['mean_median_width']:.0f}",
            flush=True,
        )
    method = choose_band_interval(candidates)
    return {"method": method, "candidates": candidates, "final": final}


def apply_interval(method: str, pred: np.ndarray, params: dict):
    return _interval_bounds(method, pred, params)


def _merge_library(base_oof: dict, added_oof: dict, libraries: list[str]) -> dict:
    """Equal-weight blend of the libraries already in `base_oof` plus one new one."""
    merged = {}
    for city, by_seed in added_oof.items():
        merged[city] = {}
        for seed, slot in by_seed.items():
            by_library = dict(base_oof[city][seed]["by_library"])
            by_library.update(slot["by_library"])
            pred = _blend(by_library, libraries, None)
            merged[city][seed] = {
                "y": base_oof[city][seed]["y"],
                "pred": pred,
                "by_library": by_library,
            }
    return merged


def _rows_from_oof(oof: dict) -> list[dict]:
    rows = []
    for city, by_seed in oof.items():
        for seed, slot in by_seed.items():
            rows.append(
                {"city": city, "seed": int(seed), **regression_metrics(slot["y"], slot["pred"])}
            )
    return rows


def select_v4(prepared: dict, *, smoke: bool = False) -> dict:
    """Development-only selection. The final holdout is not in `prepared` folds."""
    base = _base_spec()
    print("v3 specification, on the selection folds", flush=True)
    base_rows, base_oof = evaluate(prepared, base)
    base_mae = _mean_mae(base_rows)
    print(f"  v3 MAE {base_mae:.2f}", flush=True)

    kept: list[str] = []
    trials = []
    alone = {}
    alone_rows: dict[str, list] = {}
    current = base_mae
    current_oof = base_oof
    print("Sequential add-ons", flush=True)
    for name in V4_ADDON_ORDER:
        alone_spec = spec_from_kept([name])
        print(f"  alone {name}", flush=True)
        addon_rows, alone_oof = evaluate(prepared, alone_spec)
        alone_rows[name] = addon_rows
        alone[name] = _mean_mae(addon_rows)
        trial_kept = kept + [name]
        if trial_kept == [name]:
            trial_mae = alone[name]
            trial_oof = alone_oof
        else:
            print(f"  on the kept stack: {name}", flush=True)
            trial_rows, trial_oof = evaluate(prepared, spec_from_kept(trial_kept))
            trial_mae = _mean_mae(trial_rows)
        trials.append({"name": name, "mae": trial_mae, "alone_mae": alone[name]})
        if choose_quality(current, trial_mae):
            kept = trial_kept
            current = trial_mae
            current_oof = trial_oof
            print(f"  kept {name} MAE {trial_mae:.2f}", flush=True)
        else:
            print(f"  dropped {name} MAE {trial_mae:.2f} (alone {alone[name]:.2f})", flush=True)
    replay = choose_sequential(base_mae, [(item["name"], item["mae"]) for item in trials])
    if replay != kept:
        raise RuntimeError(f"Sequential replay disagreed: {replay} vs {kept}")

    spec = spec_from_kept(kept)
    libraries = list(spec["libraries"])
    point_oof = current_oof

    three_mae = None
    if not smoke:
        print("LightGBM as a third equal-weight model", flush=True)
        light_spec = spec_from_kept(kept, libraries=["lightgbm"])
        _light_rows, light_oof = evaluate(prepared, light_spec)
        trio = libraries + ["lightgbm"]
        trio_oof = _merge_library(point_oof, light_oof, trio)
        three_mae = _mean_mae(_rows_from_oof(trio_oof))
        if choose_quality(current, three_mae):
            libraries = trio
            spec = spec_from_kept(kept, libraries=libraries)
            point_oof = trio_oof
            current = three_mae
            print(f"  kept three-model blend MAE {three_mae:.2f}", flush=True)
        else:
            print(f"  dropped three-model blend MAE {three_mae:.2f}", flush=True)

    tune = {"kept": False, "name": None, "screen": [], "confirm_mae": None, "screen_default": None}
    if _has_seeds(prepared, TUNE_SCREEN_SEEDS) and _has_seeds(prepared, TUNE_CONFIRM_SEEDS):
        print("Hyperparameter screen on seeds 0–1", flush=True)
        screen_default = _mae_on(point_oof, TUNE_SCREEN_SEEDS)
        tune["screen_default"] = screen_default
        best_name = None
        best_screen = screen_default
        best_overrides = None
        for grid in TUNE_GRID:
            overrides = {key: value for key, value in grid.items() if key != "name"}
            print(f"  screen {grid['name']}", flush=True)
            screen_rows, _screen_oof = evaluate(
                prepared,
                spec_from_kept(kept, libraries=libraries, param_overrides=overrides),
                seeds=TUNE_SCREEN_SEEDS,
            )
            screen_mae = _mean_mae(screen_rows)
            tune["screen"].append({"name": grid["name"], "mae": screen_mae})
            print(f"    MAE {screen_mae:.2f} against {screen_default:.2f}", flush=True)
            if screen_mae < best_screen:
                best_screen = screen_mae
                best_name = grid["name"]
                best_overrides = overrides
        if best_name is not None and choose_confirmed(screen_default, best_screen):
            print(f"  confirm {best_name} on seeds 2–4", flush=True)
            confirm_default = _mae_on(point_oof, TUNE_CONFIRM_SEEDS)
            confirm_rows, confirm_oof = evaluate(
                prepared,
                spec_from_kept(kept, libraries=libraries, param_overrides=best_overrides),
                seeds=TUNE_CONFIRM_SEEDS,
            )
            confirm_mae = _mean_mae(confirm_rows)
            tune["confirm_mae"] = confirm_mae
            tune["confirm_default"] = confirm_default
            if choose_confirmed(confirm_default, confirm_mae):
                _screen_rows, screen_oof = evaluate(
                    prepared,
                    spec_from_kept(kept, libraries=libraries, param_overrides=best_overrides),
                    seeds=TUNE_SCREEN_SEEDS,
                )
                tuned = {city: dict(screen_oof[city]) for city in screen_oof}
                for city in confirm_oof:
                    tuned[city].update(confirm_oof[city])
                point_oof = tuned
                spec = spec_from_kept(kept, libraries=libraries, param_overrides=best_overrides)
                tune["kept"] = True
                tune["name"] = best_name
                tune["overrides"] = best_overrides
                current = _mean_mae(_rows_from_oof(point_oof))
                print(f"  kept {best_name}", flush=True)
            else:
                print(
                    f"  confirm failed MAE {confirm_mae:.2f} against {confirm_default:.2f}",
                    flush=True,
                )
        else:
            print("  no screen candidate cleared 1%", flush=True)
    else:
        tune["skipped"] = "needs seeds 0–4"
        print("  hyperparameter screen skipped (not every seed is in this run)", flush=True)

    stacker = {"kept": False, "weights": None, "confirm_mae": None}
    if _has_seeds(prepared, TUNE_SCREEN_SEEDS) and _has_seeds(prepared, TUNE_CONFIRM_SEEDS):
        print("Non-negative blend, weights fit on seeds 0–1", flush=True)
        weights = _stack_weights(point_oof, libraries, TUNE_SCREEN_SEEDS)
        weighted = _reweight(point_oof, libraries, weights)
        confirm_default = _mae_on(point_oof, TUNE_CONFIRM_SEEDS)
        confirm_mae = _mae_on(weighted, TUNE_CONFIRM_SEEDS)
        stacker["weights"] = weights
        stacker["confirm_mae"] = confirm_mae
        stacker["confirm_default"] = confirm_default
        if choose_confirmed(confirm_default, confirm_mae):
            point_oof = weighted
            stacker["kept"] = True
            current = _mean_mae(_rows_from_oof(point_oof))
            print(f"  kept stacker MAE {confirm_mae:.2f} against {confirm_default:.2f}", flush=True)
        else:
            print(
                f"  dropped stacker MAE {confirm_mae:.2f} against {confirm_default:.2f}", flush=True
            )
    else:
        stacker["skipped"] = "needs seeds 0–4"

    print("Post-hoc corrections, cross-fit", flush=True)
    uncorrected = point_oof
    correction_mae = {"none": _mean_mae(_rows_from_oof(uncorrected))}
    for method in ("smear", "isotonic"):
        rows, corrected = cross_correct(uncorrected, method)
        correction_mae[method] = _mean_mae(rows)
        if method == "smear":
            smear_oof = corrected
        else:
            isotonic_oof = corrected
        print(f"  {method} MAE {correction_mae[method]:.2f}", flush=True)
    # Smearing is one factor. Isotonic is a curve. The simpler one wins a tie.
    correction = choose_correction(
        correction_mae["none"],
        [("smear", correction_mae["smear"]), ("isotonic", correction_mae["isotonic"])],
    )
    if correction == "smear":
        point_oof = smear_oof
    elif correction == "isotonic":
        point_oof = isotonic_oof
    else:
        point_oof = uncorrected
    # The map is fit on the uncorrected out-of-fold predictions. Applying it
    # to a holdout prediction once is the same operation the cross-fit measured.
    correction_params = fit_correction_params(uncorrected, correction)
    print(f"  correction {correction}", flush=True)

    print("Interval methods, on the selection folds", flush=True)
    interval = calibrate_intervals(point_oof)
    print(f"  interval {interval['method']}", flush=True)
    selected_rows = _rows_from_oof(point_oof)
    return {
        "kept": kept,
        "base_mae": base_mae,
        "trials": trials,
        "alone": alone,
        "alone_rows": alone_rows,
        "selected_mae": _mean_mae(selected_rows),
        "rows": selected_rows,
        "base_rows": base_rows,
        "libraries": libraries,
        "three_library_mae": three_mae,
        "tune": tune,
        "stacker": stacker,
        "correction": correction,
        "correction_mae": correction_mae,
        "correction_params": correction_params,
        "spec": spec,
        "interval_method": interval["method"],
        "interval_candidates": interval["candidates"],
        "interval_final": interval["final"],
        "feature_set": _feature_label(kept, libraries, tune, stacker, correction),
    }


def _feature_label(kept, libraries, tune, stacker, correction) -> str:
    parts = ["v3", *kept]
    if len(libraries) > 2:
        parts.append("lightgbm")
    if tune.get("kept"):
        parts.append(str(tune["name"]))
    if stacker.get("kept"):
        parts.append("stack")
    if correction != "none":
        parts.append(correction)
    return "+".join(parts)


def fit_holdout(frame, extra, train_idx, eval_idx, spec, weights_for_city):
    """Refit on `train_idx` plus any allowed historical rows, and score `eval_idx`.

    Early stopping picks the tree count, then the model is retrained on every
    training row. The historical filter treats `eval_idx` as the blocked set.
    """
    X_train, y_log, X_eval, y_eval, sample_weight, groups, state = build_matrices(
        frame, extra, train_idx, eval_idx, np.array([], dtype=int), spec
    )
    by_library, models, tree_counts = _fit_libraries(
        X_train, y_log, groups, sample_weight, X_eval, spec, refit=True
    )
    pred = _blend(by_library, spec["libraries"], weights_for_city)
    return {
        "pred": pred,
        "y": y_eval,
        "models": models,
        "tree_counts": tree_counts,
        "state": state,
        "columns": list(X_eval.columns),
        "matrix_valid": X_eval,
        "n_extra": int(state["n_extra"]),
        "by_library": by_library,
    }
