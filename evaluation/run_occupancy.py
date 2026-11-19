"""Fit the nights model and score the seed-31 holdout once.

    python -m evaluation.run_occupancy

The holdout file is written before any fit, from coordinates and the seed.
Feature choice uses the development folds only. The holdout is not an input
to that choice. This does not refit the published price artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from data.cities import CITIES
from data.ingest import load_study_frame
from data.occupancy import ANNUAL_NIGHTS_CAP
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from evaluation.audit import split_audit
from evaluation.blocking import development_splits, score_split
from evaluation.fitting import (
    design_matrices,
    fit_boosted,
    params_for_depth,
    predict_dollars,
    predict_log,
    refit_boosted,
)
from evaluation.freeze import listing_ids
from evaluation.metrics import regression_metrics
from evaluation.occupancy_freeze import ensure_occupancy_ids, occupancy_split, population
from evaluation.occupancy_protocol import (
    CLASS_ACCURACY_MARGIN,
    FEATURE_LADDER,
    GRADE_QUANTILES,
    INNER_SEEDS,
    MAE_RATIO_BAR,
    OCCUPANCY_DEPTH,
    OCCUPANCY_SEED,
    R2_BAR,
    assert_columns_allowed,
    cleared_classification,
    cleared_regression,
    ladder_columns,
)
from evaluation.protocol import V2_BLEND, V2_DEPTH, study_cities, v3_features
from evaluation.select import choose_blend, choose_quality, choose_simpler
from evaluation.splits import spatial_group_ids

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
ARTIFACTS = REPO / "models" / "artifacts"
CLASS_NAMES = ("Low", "Medium", "High")


def _as_frame(frame: pd.DataFrame, idx: np.ndarray, columns: list[str]) -> pd.DataFrame:
    matrix = frame.iloc[np.asarray(idx)][list(columns)].apply(pd.to_numeric, errors="coerce")
    return matrix.reset_index(drop=True)


def _nights_from(model, matrix: pd.DataFrame, library: str, logged: bool) -> np.ndarray:
    raw = predict_log(model, matrix, library)
    if logged:
        raw = np.expm1(raw)
    return np.clip(np.maximum(raw, 0.0), 0.0, ANNUAL_NIGHTS_CAP)


def _fit_one(library, X_train, y_train, groups, X_eval, logged: bool, *, refit: bool):
    target = np.log1p(y_train) if logged else np.asarray(y_train, dtype=float)
    params = params_for_depth(OCCUPANCY_DEPTH)
    stopped = fit_boosted(library, params, X_train, target, groups)
    trees = int(getattr(stopped, "_capscout_trees", None) or 200)
    model = stopped
    if refit:
        model = refit_boosted(library, params, trees, X_train, target)
    return model, _nights_from(model, X_eval, library, logged), trees


def _blend(preds: list[np.ndarray]) -> np.ndarray:
    return np.clip(np.mean(np.vstack(preds), axis=0), 0.0, ANNUAL_NIGHTS_CAP)


def _area_median(
    neighbourhood: np.ndarray, y: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray
) -> tuple[np.ndarray, dict[str, float], float]:
    y_train = np.asarray(y, dtype=float)[train_idx]
    default = float(np.median(y_train))
    keys = np.asarray(neighbourhood).astype(str)
    mapping = (
        pd.Series(y_train)
        .groupby(pd.Series(keys[train_idx]), sort=False)
        .median()
        .astype(float)
        .to_dict()
    )
    pred = np.array([mapping.get(keys[i], default) for i in eval_idx], dtype=float)
    return pred, {str(key): float(value) for key, value in mapping.items()}, default


def _room_median(frame: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray):
    from evaluation.fitting import median_baseline

    return median_baseline(
        frame["neighbourhood_cleansed"].to_numpy(),
        frame["room_type"].to_numpy(),
        frame["bedrooms"].to_numpy(dtype=float),
        y,
        train_idx,
        eval_idx,
        by_bedroom=False,
    )


def _city_mean(rows: list[dict], key: str = "mae") -> float:
    cities = sorted({row["city"] for row in rows})
    if not cities:
        raise RuntimeError("no development rows")
    return float(
        np.mean([np.mean([row[key] for row in rows if row["city"] == city]) for city in cities])
    )


def _mean_std(rows: list[dict], city: str, key: str) -> dict[str, float]:
    values = np.array([row[key] for row in rows if row["city"] == city], dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(values.mean()), "std": std, "n_seeds": int(len(values))}


def _class_labels(y: np.ndarray, low: float, high: float) -> np.ndarray:
    labels = np.ones(len(y), dtype=int)
    labels[np.asarray(y) <= low] = 0
    labels[np.asarray(y) > high] = 2
    return labels


def _majority(neighbourhood: np.ndarray, labels: np.ndarray, train_idx, eval_idx) -> np.ndarray:
    train_labels = labels[train_idx]
    values, counts = np.unique(train_labels, return_counts=True)
    city_majority = int(values[int(np.argmax(counts))])
    keys = np.asarray(neighbourhood).astype(str)
    buckets: dict[str, list[int]] = {}
    for key, label in zip(keys[train_idx], train_labels, strict=True):
        buckets.setdefault(str(key), []).append(int(label))
    mapping = {}
    for key, bucket in buckets.items():
        vals, cts = np.unique(bucket, return_counts=True)
        mapping[key] = int(vals[int(np.argmax(cts))])
    return np.array([mapping.get(keys[i], city_majority) for i in eval_idx], dtype=int)


def _new_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=OCCUPANCY_DEPTH,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


def _classification_report(y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    labels = [0, 1, 2]
    matrix = confusion_matrix(y_true, pred, labels=labels)
    reliability = []
    for index, name in enumerate(CLASS_NAMES):
        mask = pred == index
        reliability.append(
            {
                "class": name,
                "n": int(mask.sum()),
                "empirical_frequency": (
                    float((y_true[mask] == index).mean()) if mask.any() else None
                ),
                "mean_predicted_probability": (
                    float(proba[mask, index].mean()) if mask.any() else None
                ),
            }
        )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "confusion": matrix.astype(int).tolist(),
        "reliability": reliability,
    }


def _grade_cuts(revenue: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(np.asarray(revenue, dtype=float), GRADE_QUANTILES)
    keys = ("p20", "p40", "p60", "p80")
    return {key: round(float(value), 2) for key, value in zip(keys, quantiles, strict=True)}


def _assign_grade(revenue: np.ndarray, cuts: dict[str, float]) -> list[str]:
    from models.grade import grade_revenue

    return [str(grade_revenue(value, cuts)["letter"]) for value in revenue]


def _metrics_block(y_true, y_pred) -> dict:
    block = regression_metrics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    return {key: block[key] for key in ("r2", "mae", "rmse", "mape", "median_ape", "n")}


def _price_on_holdout(full: pd.DataFrame, holdout_idx: np.ndarray) -> np.ndarray:
    """Published price blend, refit so the occupancy holdout is not in training."""
    lat = full["latitude"].to_numpy(dtype=float)
    lon = full["longitude"].to_numpy(dtype=float)
    hosts = full["host_id"].to_numpy()
    train_idx, _info = score_split(lat, lon, hosts, holdout_idx)
    columns, spatial, text = v3_features()
    if spatial != "none" or text != "none":
        raise RuntimeError("The published price model is no longer a static column list.")
    X_train, y_log, X_eval, _y_price, _state = design_matrices(
        full, train_idx, holdout_idx, columns, "none"
    )
    groups = spatial_group_ids(lat, lon)[train_idx]
    preds = []
    for library in V2_BLEND:
        params = params_for_depth(V2_DEPTH)
        stopped = fit_boosted(library, params, X_train, y_log, groups)
        trees = int(getattr(stopped, "_capscout_trees", None) or 80)
        model = refit_boosted(library, params, trees, X_train, y_log)
        preds.append(predict_dollars(model, X_eval, library))
    return np.mean(np.vstack(preds), axis=0)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None:
        return None
    return value


def _prepare(city: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    full, stats = load_study_frame(city)
    observed = population(full)
    split = occupancy_split(
        observed["latitude"].to_numpy(dtype=float),
        observed["longitude"].to_numpy(dtype=float),
        observed["host_id"].to_numpy(),
    )
    return full, observed, {"split": split, "stats": stats}


def _score_prefix(observed, split, columns, logged: bool, libraries: tuple[str, ...], tag: str):
    lat = observed["latitude"].to_numpy(dtype=float)
    lon = observed["longitude"].to_numpy(dtype=float)
    hosts = observed["host_id"].to_numpy()
    y = observed["booked_nights"].to_numpy(dtype=float)
    rows = []
    for seed, train_idx, valid_idx, _info in development_splits(
        split["groups"],
        split["development_idx"],
        lat,
        lon,
        hosts,
        seeds=INNER_SEEDS,
    ):
        X_train = _as_frame(observed, train_idx, columns)
        X_valid = _as_frame(observed, valid_idx, columns)
        groups = split["groups"][train_idx]
        y_train = y[train_idx]
        preds = []
        for library in libraries:
            _model, pred, _trees = _fit_one(
                library, X_train, y_train, groups, X_valid, logged, refit=False
            )
            preds.append(pred)
        pred = _blend(preds)
        metrics = regression_metrics(y[valid_idx], pred)
        rows.append({"tag": tag, "city": "", "seed": int(seed), **metrics})
    return rows


def _fit_development(prepared: dict) -> dict:
    """Choose the prefix, the transform, and whether the blend earns its place."""
    records: list[dict] = []
    print("Selecting feature prefixes on development folds.", flush=True)
    for name, columns in FEATURE_LADDER:
        assert_columns_allowed(columns)
        print(f"  prefix {name}", flush=True)
        for city, bundle in prepared.items():
            observed = bundle["observed"]
            rows = _score_prefix(observed, bundle["split"], columns, True, ("xgboost",), name)
            for row in rows:
                row["city"] = city
                row["tag"] = name
            records.extend(rows)
    ladder = [
        (name, _city_mean([row for row in records if row["tag"] == name]))
        for name, _cols in FEATURE_LADDER
    ]
    chosen = choose_simpler(ladder)
    chosen_columns = ladder_columns(chosen)
    log_mae = dict(ladder)[chosen]
    print(
        f"Chosen prefix {chosen} at development MAE {log_mae:.3f}. Comparing the raw target.",
        flush=True,
    )
    raw_rows: list[dict] = []
    for city, bundle in prepared.items():
        rows = _score_prefix(
            bundle["observed"], bundle["split"], chosen_columns, False, ("xgboost",), "raw"
        )
        for row in rows:
            row["city"] = city
        raw_rows.extend(rows)
    raw_mae = _city_mean(raw_rows)
    logged = not choose_quality(log_mae, raw_mae)
    xgb_mae = log_mae if logged else raw_mae
    print(f"Target is {'log1p' if logged else 'raw nights'}. Fitting the blend.", flush=True)
    blend_rows: list[dict] = []
    for city, bundle in prepared.items():
        rows = _score_prefix(
            bundle["observed"],
            bundle["split"],
            chosen_columns,
            logged,
            ("catboost", "xgboost"),
            "blend",
        )
        for row in rows:
            row["city"] = city
        blend_rows.extend(rows)
    blend_mae = _city_mean(blend_rows)
    use_blend = choose_blend(xgb_mae, blend_mae)
    libraries = ("catboost", "xgboost") if use_blend else ("xgboost",)
    if use_blend:
        kept_rows = blend_rows
    elif logged:
        kept_rows = [row for row in records if row["tag"] == chosen]
    else:
        kept_rows = raw_rows
    return {
        "records": records,
        "ladder": ladder,
        "chosen": chosen,
        "columns": chosen_columns,
        "logged": logged,
        "log_mae": log_mae,
        "raw_mae": raw_mae,
        "xgb_mae": xgb_mae,
        "blend_mae": blend_mae,
        "use_blend": use_blend,
        "libraries": libraries,
        "kept_rows": kept_rows,
    }


def _holdout_models(observed, split, columns, logged, libraries):
    train_idx = split["development_idx"]
    test_idx = split["test_idx"]
    y = observed["booked_nights"].to_numpy(dtype=float)
    X_train = _as_frame(observed, train_idx, columns)
    X_test = _as_frame(observed, test_idx, columns)
    groups = split["groups"][train_idx]
    models = {}
    preds = []
    trees = {}
    for library in libraries:
        model, pred, n_trees = _fit_one(
            library, X_train, y[train_idx], groups, X_test, logged, refit=True
        )
        models[library] = model
        preds.append(pred)
        trees[library] = n_trees
    return models, _blend(preds), trees, train_idx, test_idx


def _classifier(observed, split, columns):
    y = observed["booked_nights"].to_numpy(dtype=float)
    train_idx = split["development_idx"]
    low, high = np.quantile(y[train_idx], [1.0 / 3.0, 2.0 / 3.0])
    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    labels = _class_labels(y, low, high)
    X_train = _as_frame(observed, train_idx, columns)
    model = _new_classifier()
    model.fit(X_train, labels[train_idx])
    return {"model": model, "low": low, "high": high, "labels": labels}


def _score_classifier(bundle, clf, test_idx) -> dict:
    observed = bundle["observed"]
    X_test = _as_frame(observed, test_idx, bundle["columns"])
    proba = clf["model"].predict_proba(X_test)
    # Columns follow model.classes_, which are 0, 1, 2 when every class was fit.
    class_index = {int(label): pos for pos, label in enumerate(clf["model"].classes_)}
    ordered = np.column_stack([proba[:, class_index[label]] for label in (0, 1, 2)])
    pred = np.argmax(ordered, axis=1).astype(int)
    y_true = clf["labels"][test_idx]
    report = _classification_report(y_true, pred, ordered)
    baseline = _majority(
        observed["neighbourhood_cleansed"].to_numpy(),
        clf["labels"],
        bundle["split"]["development_idx"],
        test_idx,
    )
    report["baseline_accuracy"] = float(accuracy_score(y_true, baseline))
    report["baseline_balanced_accuracy"] = float(balanced_accuracy_score(y_true, baseline))
    report["cleared_bar"] = cleared_classification(
        report["accuracy"],
        report["baseline_accuracy"],
        report["balanced_accuracy"],
        report["baseline_balanced_accuracy"],
    )
    report["cuts"] = {"low_max": clf["low"], "high_min": clf["high"]}
    report["pred"] = pred
    report["baseline_pred"] = baseline
    return report


def _leakage_demo(observed, split, columns) -> dict:
    """One development seed, with the review count added. Not a candidate."""
    lat = observed["latitude"].to_numpy(dtype=float)
    lon = observed["longitude"].to_numpy(dtype=float)
    hosts = observed["host_id"].to_numpy()
    y = observed["booked_nights"].to_numpy(dtype=float)
    folded = list(
        development_splits(split["groups"], split["development_idx"], lat, lon, hosts, seeds=(0,))
    )
    if not folded:
        return {"seed": 0, "note": "seed 0 was too small to score"}
    _seed, train_idx, valid_idx, _info = folded[0]
    bare = _as_frame(observed, train_idx, columns)
    bare_valid = _as_frame(observed, valid_idx, columns)
    groups = split["groups"][train_idx]
    _model, plain, _trees = _fit_one(
        "xgboost", bare, y[train_idx], groups, bare_valid, True, refit=False
    )
    leaked_columns = list(columns) + ["number_of_reviews_ltm"]
    leaked = _as_frame(observed, train_idx, leaked_columns)
    leaked_valid = _as_frame(observed, valid_idx, leaked_columns)
    _model, revealed, _trees = _fit_one(
        "xgboost", leaked, y[train_idx], groups, leaked_valid, True, refit=False
    )
    return {
        "seed": 0,
        "without_reviews_mae": regression_metrics(y[valid_idx], plain)["mae"],
        "with_reviews_ltm_mae": regression_metrics(y[valid_idx], revealed)["mae"],
        "note": (
            "number_of_reviews_ltm is an input to the label. This fit is a "
            "demonstration that the column reconstructs the target. It is not eligible."
        ),
    }


def _label_summary(observed: pd.DataFrame, split: dict) -> dict:
    nights = observed["booked_nights"].to_numpy(dtype=float)
    return {
        "n_observed": int(len(observed)),
        "n_train": int(len(split["development_idx"])),
        "n_test": int(len(split["test_idx"])),
        "median_nights": float(np.median(nights)),
        "mean_nights": float(np.mean(nights)),
        "share_at_cap": float(np.mean(np.isclose(nights, ANNUAL_NIGHTS_CAP))),
        "train_removed_host": int(split["info"].get("removed_host", 0)),
        "train_removed_buffer": int(split["info"].get("removed_buffer", 0)),
    }


def main() -> None:
    cities = study_cities()
    frozen = ensure_occupancy_ids(cities)
    print(f"Occupancy holdout seed {OCCUPANCY_SEED} is on disk. Fitting starts now.", flush=True)
    prepared = {}
    full_frames = {}
    for city in cities:
        full, observed, bundle = _prepare(city)
        if observed["id"].duplicated().any():
            raise RuntimeError(f"{city} occupancy population has duplicate listing ids")
        test_ids = listing_ids(observed.iloc[bundle["split"]["test_idx"]]["id"])
        if sorted(test_ids, key=int) != frozen["cities"][city]:
            raise RuntimeError(f"{city} split does not match the frozen occupancy ids")
        audit = split_audit(
            observed, bundle["split"]["development_idx"], bundle["split"]["test_idx"]
        )
        if not audit["passed"]:
            raise RuntimeError(f"{city} occupancy split failed the audit: {audit}")
        bundle["audit"] = audit
        bundle["observed"] = observed
        prepared[city] = bundle
        full_frames[city] = full
        summary = _label_summary(observed, bundle["split"])
        print(
            f"{city}: {summary['n_observed']} observed, {summary['n_test']} held out, "
            f"{summary['share_at_cap']:.1%} at the cap.",
            flush=True,
        )

    selection = _fit_development(prepared)
    columns = selection["columns"]
    logged = selection["logged"]
    libraries = selection["libraries"]
    print(
        f"Holdout model: {selection['chosen']}, "
        f"{'log' if logged else 'raw'}, {', '.join(libraries)}.",
        flush=True,
    )

    predictions = []
    city_reports = {}
    leakage = {}
    for city in cities:
        print(f"Scoring {city} holdout.", flush=True)
        bundle = prepared[city]
        observed = bundle["observed"]
        split = bundle["split"]
        models, pred_nights, trees, train_idx, test_idx = _holdout_models(
            observed, split, columns, logged, libraries
        )
        y = observed["booked_nights"].to_numpy(dtype=float)
        price = observed["price_clean"].to_numpy(dtype=float)
        area_pred, area_map, city_median = _area_median(
            observed["neighbourhood_cleansed"].to_numpy(), y, train_idx, test_idx
        )
        room_pred = _room_median(observed, y, train_idx, test_idx)
        model_metrics = _metrics_block(y[test_idx], pred_nights)
        area_metrics = _metrics_block(y[test_idx], area_pred)
        room_metrics = _metrics_block(y[test_idx], room_pred)
        cleared = cleared_regression(model_metrics["mae"], area_metrics["mae"], model_metrics["r2"])
        reduction = (
            float((area_metrics["mae"] - model_metrics["mae"]) / area_metrics["mae"])
            if area_metrics["mae"]
            else None
        )

        full = full_frames[city]
        id_str = pd.to_numeric(full["id"], errors="raise").astype("int64").astype(str)
        test_id_list = listing_ids(observed.iloc[test_idx]["id"])
        holdout_positions = []
        position_of = {value: index for index, value in enumerate(id_str.tolist())}
        for listing_id in test_id_list:
            if listing_id not in position_of:
                raise RuntimeError(f"{listing_id} missing from the {city} price frame")
            holdout_positions.append(position_of[listing_id])
        holdout_idx = np.asarray(holdout_positions, dtype=int)
        print(f"  refitting the published price model for {city} revenue.", flush=True)
        pred_price = _price_on_holdout(full, holdout_idx)
        y_revenue = price[test_idx] * y[test_idx]
        pred_revenue = pred_price * pred_nights
        revenue_metrics = _metrics_block(y_revenue, pred_revenue)
        price_only = _metrics_block(y_revenue, pred_price * y[test_idx])
        nights_only = _metrics_block(y_revenue, price[test_idx] * pred_nights)

        train_revenue = price[train_idx] * y[train_idx]
        cuts = _grade_cuts(train_revenue)
        grades = _assign_grade(pred_revenue, cuts)
        actual_grades = _assign_grade(y_revenue, cuts)

        print(f"  classifying {city}.", flush=True)
        clf = _classifier(observed, split, columns)
        bundle["columns"] = columns
        class_report = None
        pred_class = np.array([""] * len(test_idx))
        if clf is not None:
            class_report = _score_classifier(bundle, clf, test_idx)
            pred_class = np.array([CLASS_NAMES[int(i)] for i in class_report.pop("pred")])
            class_report.pop("baseline_pred")

        dev_rows = [row for row in selection["kept_rows"] if row["city"] == city]
        area_dev = []
        for seed, tr, va, _info in development_splits(
            split["groups"],
            split["development_idx"],
            observed["latitude"].to_numpy(dtype=float),
            observed["longitude"].to_numpy(dtype=float),
            observed["host_id"].to_numpy(),
            seeds=INNER_SEEDS,
        ):
            baseline, _map, _default = _area_median(
                observed["neighbourhood_cleansed"].to_numpy(), y, tr, va
            )
            area_dev.append(
                {"city": city, "seed": int(seed), **regression_metrics(y[va], baseline)}
            )

        artifact = {
            "kind": "occupancy",
            "city": city,
            "seed": OCCUPANCY_SEED,
            "columns": columns,
            "logged": logged,
            "depth": OCCUPANCY_DEPTH,
            "libraries": list(libraries),
            "models": models,
            "tree_counts": trees,
            "annual_cap": ANNUAL_NIGHTS_CAP,
            "grade_cuts": cuts,
            "area_medians": area_map,
            "city_median_nights": city_median,
            "classifier": None if clf is None else clf["model"],
            "class_cuts": None if clf is None else {"low_max": clf["low"], "high_min": clf["high"]},
            "class_names": list(CLASS_NAMES),
            "holdout_mae": model_metrics["mae"],
            "baseline_mae": area_metrics["mae"],
            "cleared_bar": cleared,
            "classification_cleared": bool(class_report and class_report["cleared_bar"]),
        }
        path = ARTIFACTS / city / "occupancy.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path, compress=3)

        city_reports[city] = {
            "display": CITIES[city]["display"],
            "label": _label_summary(observed, split),
            "audit": bundle["audit"],
            "grade_cuts": cuts,
            "development": {
                "model": {
                    "mae": _mean_std(dev_rows, city, "mae"),
                    "r2": _mean_std(dev_rows, city, "r2"),
                },
                "baseline_area": {
                    "mae": _mean_std(area_dev, city, "mae"),
                    "r2": _mean_std(area_dev, city, "r2"),
                },
            },
            "holdout": {
                "model": model_metrics,
                "baseline_area": area_metrics,
                "baseline_room": room_metrics,
                "mae_reduction_vs_area": reduction,
                "cleared_bar": cleared,
                "revenue": revenue_metrics,
                "revenue_if_nights_known": price_only,
                "revenue_if_price_known": nights_only,
                "grade_match": float(np.mean(np.array(grades) == np.array(actual_grades))),
                "classification": {
                    key: value
                    for key, value in (class_report or {}).items()
                    if key not in {"pred", "baseline_pred"}
                },
            },
        }
        leakage[city] = _leakage_demo(observed, split, columns)
        frame = pd.DataFrame(
            {
                "city": city,
                "id": test_id_list,
                "y_nights": y[test_idx],
                "pred_nights": pred_nights,
                "baseline_area": area_pred,
                "baseline_room": room_pred,
                "price": price[test_idx],
                "pred_price": pred_price,
                "y_revenue": y_revenue,
                "pred_revenue": pred_revenue,
                "grade": grades,
                "grade_actual": actual_grades,
                "pred_class": pred_class,
            }
        )
        predictions.append(frame)
        print(
            f"  {city} R² {model_metrics['r2']:.3f} MAE {model_metrics['mae']:.1f} "
            f"vs area {area_metrics['mae']:.1f} cleared={cleared}",
            flush=True,
        )

    payload = {
        "smoke": False,
        "seed": OCCUPANCY_SEED,
        "target": {
            "name": "booked_nights_ltm",
            "review_rate": 0.50,
            "average_stay_nights": 3.0,
            "cap_nights": round(float(ANNUAL_NIGHTS_CAP), 1),
            "minimum_nights_rule": (
                "Use the listing minimum when it is longer than 3 nights and at most 90. "
                "Otherwise use 3 nights. The minimum is part of the label and is not a feature."
            ),
            "unobserved": "Zero or missing reviews in the last 12 months are left out, not coded as zero.",
            "source": "Inside Airbnb San Francisco model, on number_of_reviews_ltm, annualised.",
        },
        "bar": {
            "mae_ratio": MAE_RATIO_BAR,
            "r2": R2_BAR,
            "class_accuracy_margin": CLASS_ACCURACY_MARGIN,
            "rule": (
                "A city clears the regression bar when holdout MAE is at most 90% of the "
                "neighbourhood-median MAE and holdout R² is at least 0.10. Classification "
                "clears when accuracy beats the neighbourhood majority class by 8 points "
                "and balanced accuracy is higher than that baseline."
            ),
        },
        "protocol": {
            "feature_set": selection["chosen"],
            "logged": selection["logged"],
            "blend": selection["use_blend"],
            "libraries": list(selection["libraries"]),
            "depth": OCCUPANCY_DEPTH,
            "ladder_development_mae": {name: mae for name, mae in selection["ladder"]},
            "log_development_mae": selection["log_mae"],
            "raw_development_mae": selection["raw_mae"],
            "xgboost_development_mae": selection["xgb_mae"],
            "blend_development_mae": selection["blend_mae"],
            "columns": columns,
        },
        "cities": city_reports,
        "leakage_demo": leakage,
        "cleared_all_cities": all(city_reports[city]["holdout"]["cleared_bar"] for city in cities),
        "classification_cleared_all_cities": all(
            city_reports[city]["holdout"]["classification"].get("cleared_bar") for city in cities
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "occupancy_metrics.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    pd.concat(predictions, ignore_index=True).to_csv(
        REPORTS / "occupancy_predictions.csv", index=False
    )
    print("Wrote reports/occupancy_metrics.json", flush=True)


if __name__ == "__main__":
    main()
