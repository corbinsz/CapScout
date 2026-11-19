"""Fit the simple-model study. The holdout file is written before any fit.

python -m evaluation.run_study
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from data.cities import CITIES, STUDY_CITIES
from data.clean import render_quality_markdown
from data.occupancy import ANNUAL_NIGHTS_CAP
from sklearn.model_selection import GroupKFold

from evaluation.metrics import interval_metrics, regression_metrics
from evaluation.study_data import prepare_city
from evaluation.study_protocol import (
    BOOTSTRAP_SEED,
    ELIGIBLE_MODELS,
    FEATURE_MARGIN,
    INTERVAL_MIN_BIN,
    MAX_MISSING_FRACTION,
    N_BOOTSTRAP,
    N_FOLDS,
    NIGHTS_FORBIDDEN,
    NIGHTS_LADDER,
    PRICE_FORBIDDEN,
    PRICE_LADDER,
    assert_seeds,
    challenger_reason,
    choose_nights_target,
    choose_params,
    choose_shipped,
    choose_simplest,
    nights_cleared,
    nights_reason,
    nights_selection_reason,
    selection_reason,
)
from evaluation.study_split import PATH as ID_PATH
from evaluation.study_split import audit_split, ensure_ids, split_frame

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2) + "\n")


def _folds(frame: pd.DataFrame, train_idx: np.ndarray) -> tuple[list, int]:
    train_idx = np.asarray(train_idx, dtype=int)
    groups = frame["cell"].to_numpy()[train_idx]
    n_groups = int(len(np.unique(groups)))
    n_splits = min(N_FOLDS, n_groups)
    if n_splits < 2:
        raise RuntimeError(f"only {n_groups} training cells")
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_pos, valid_pos in splitter.split(np.zeros(len(train_idx)), groups=groups):
        folds.append((train_idx[train_pos], train_idx[valid_pos]))
    covered = np.concatenate([valid for _train, valid in folds])
    if len(covered) != len(train_idx) or set(covered.tolist()) != set(train_idx.tolist()):
        raise RuntimeError("grouped folds did not cover the training rows once")
    return folds, n_splits


def _mean(by_city: dict, name: str) -> float:
    return float(np.mean([by_city[city][name] for city in STUDY_CITIES]))


def _mean_mae(block: dict, name: str) -> float:
    return float(np.mean([block[city][name]["mae"] for city in STUDY_CITIES]))


def _assert_allowed(columns: list[str], forbidden: frozenset) -> None:
    bad = sorted(set(columns) & set(forbidden))
    if bad:
        raise RuntimeError(f"forbidden columns in the model: {bad}")


def _prune(frames: dict[str, pd.DataFrame], columns: list[str]) -> tuple[list[str], list[dict]]:
    kept, dropped = [], []
    for column in columns:
        missing = {city: float(frame[column].isna().mean()) for city, frame in frames.items()}
        if float(np.mean(list(missing.values()))) > MAX_MISSING_FRACTION:
            dropped.append({"feature": column, "missing": missing})
        else:
            if any(column not in frame.columns for frame in frames.values()):
                raise RuntimeError(f"missing column {column}")
            kept.append(column)
    return kept, dropped


def _prune_ladder(frames, ladder):
    pruned = []
    dropped = []
    for name, columns in ladder:
        kept, gone = _prune(frames, list(columns))
        dropped.extend(gone)
        if pruned and pruned[-1][1] == kept:
            continue
        pruned.append((name, kept))
    seen = set()
    unique = []
    for item in dropped:
        if item["feature"] in seen:
            continue
        seen.add(item["feature"])
        unique.append(item)
    if not pruned:
        raise RuntimeError("every feature was dropped")
    return pruned, unique


def _metrics_at(y, pred, idx) -> dict:
    return regression_metrics(np.asarray(y)[idx], np.asarray(pred)[idx])


def run() -> dict:
    assert_seeds()
    print("Loading and cleaning", flush=True)
    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for city in STUDY_CITIES:
        frames[city], reports[city] = prepare_city(city)
        print(
            f"  {city}: {reports[city]['rows_raw']:,} -> {reports[city]['rows_final']:,} rows, "
            f"{reports[city]['n_cells']} cells",
            flush=True,
        )

    print(f"Freezing holdout ids at {ID_PATH.name}", flush=True)
    frozen = ensure_ids(frames)
    if "price" in json.dumps(frozen["cities"]):
        raise RuntimeError("the holdout file contains a price field")
    print("Holdout frozen. No model has been fit.", flush=True)
    quality_json = REPORTS / "data_quality.json"
    quality_md = REPORTS / "data_quality.md"
    _write(quality_json, {"cities": reports})
    quality_md.write_text(render_quality_markdown(reports))

    # Imports that pull in the estimators happen after the ids are on disk.
    from models.artifacts import save_artifact
    from models.grade import grade_revenue

    from evaluation.study_models import (
        AreaMedian,
        BlendModel,
        apply_interval,
        bootstrap_metrics,
        cv_area,
        cv_mae,
        direction_checks,
        dollar_effects,
        fit_interval,
        make_catboost,
        make_forest,
        make_linear,
        make_network,
        make_ridge,
        make_xgb,
        oof_predict,
        shap_means,
    )

    splits = {}
    for city in STUDY_CITIES:
        split = split_frame(frames[city], city, frozen["primary_seed"])
        folds, n_splits = _folds(frames[city], split["train_idx"])
        audit = audit_split(frames[city], split["train_idx"], split["test_idx"])
        if not audit["passed"]:
            raise RuntimeError(f"{city} primary split failed the audit: {audit}")
        splits[city] = {**split, "folds": folds, "n_splits": n_splits, "audit": audit}
        info = split["info"]
        print(
            f"  {city} primary: test {info['n_test']:,} in {info['n_test_cells']} cells, "
            f"train {info['train_after']:,} "
            f"(dropped {info['removed_host']:,} host, {info['removed_buffer']:,} buffer)",
            flush=True,
        )

    price_ladder, price_dropped = _prune_ladder(frames, PRICE_LADDER)
    nights_ladder, nights_dropped = _prune_ladder(frames, NIGHTS_LADDER)
    for _name, columns in price_ladder:
        _assert_allowed(columns, PRICE_FORBIDDEN)
    for _name, columns in nights_ladder:
        _assert_allowed(columns, NIGHTS_FORBIDDEN)

    print("Price feature prefixes, grouped CV, fixed XGBoost", flush=True)
    price_feature_by_city = {city: {} for city in STUDY_CITIES}
    for name, columns in price_ladder:

        def _price_probe(columns=columns):
            return make_xgb(columns, 4, 8, kind="log", high=None)

        for city in STUDY_CITIES:
            y = frames[city]["price_clean"].to_numpy(dtype=float)
            mae = cv_mae(_price_probe, frames[city], y, splits[city]["folds"])
            price_feature_by_city[city][name] = mae
            print(f"  {city} {name} MAE {mae:.2f}", flush=True)
    price_feature_mean = {
        name: _mean(price_feature_by_city, name) for name, _columns in price_ladder
    }
    price_feature_name = choose_simplest(
        [(name, price_feature_mean[name]) for name, _columns in price_ladder],
        FEATURE_MARGIN,
    )
    price_columns = dict(price_ladder)[price_feature_name]
    print(f"Price features: {price_feature_name} ({len(price_columns)} columns)", flush=True)

    print("Nights feature prefixes, grouped CV, direct nights", flush=True)
    nights_feature_by_city = {city: {} for city in STUDY_CITIES}
    for name, columns in nights_ladder:

        def _nights_probe(columns=columns):
            return make_xgb(columns, 4, 8, kind="dollar", high=ANNUAL_NIGHTS_CAP)

        for city in STUDY_CITIES:
            y = frames[city]["booked_nights"].to_numpy(dtype=float)
            mae = cv_mae(_nights_probe, frames[city], y, splits[city]["folds"])
            nights_feature_by_city[city][name] = mae
            print(f"  {city} {name} MAE {mae:.2f}", flush=True)
    nights_feature_mean = {
        name: _mean(nights_feature_by_city, name) for name, _columns in nights_ladder
    }
    nights_feature_name = choose_simplest(
        [(name, nights_feature_mean[name]) for name, _columns in nights_ladder],
        FEATURE_MARGIN,
    )
    nights_columns = dict(nights_ladder)[nights_feature_name]
    print(f"Nights features: {nights_feature_name} ({len(nights_columns)} columns)", flush=True)

    print("Nights target: direct versus log, on the chosen features", flush=True)
    raw_by_city, log_by_city = {}, {}
    for city in STUDY_CITIES:
        y = frames[city]["booked_nights"].to_numpy(dtype=float)
        folds = splits[city]["folds"]

        def _raw():
            return make_xgb(nights_columns, 4, 8, kind="dollar", high=ANNUAL_NIGHTS_CAP)

        def _logged():
            return make_xgb(nights_columns, 4, 8, kind="log", high=ANNUAL_NIGHTS_CAP)

        raw_by_city[city] = cv_mae(_raw, frames[city], y, folds)
        log_by_city[city] = cv_mae(_logged, frames[city], y, folds)
        print(f"  {city} raw {raw_by_city[city]:.2f} log {log_by_city[city]:.2f}", flush=True)
    raw_mae = float(np.mean(list(raw_by_city.values())))
    log_mae = float(np.mean(list(log_by_city.values())))
    nights_target = choose_nights_target(raw_mae, log_mae)
    nights_kind = "log" if nights_target == "log" else "dollar"
    if nights_target == "raw":
        nights_target_sentence = (
            "The nights models predict nights directly. On the training folds a log target "
            f"averaged {log_mae:.2f} nights of error against {raw_mae:.2f} for the direct "
            "target, which is not a 2% improvement, so the log target was not used."
        )
    else:
        nights_target_sentence = (
            "On the training folds a log target cut average error from "
            f"{raw_mae:.2f} nights to {log_mae:.2f}, which clears 2%. The nights models "
            "predict log nights and are converted back to nights before scoring."
        )
    print(nights_target_sentence, flush=True)

    def tune_and_score(
        y_column: str, columns: list[str], kind: str, high: float | None, label: str
    ):
        tuned = {}
        predictions = {}
        for city in STUDY_CITIES:
            frame = frames[city]
            y = frame[y_column].to_numpy(dtype=float)
            folds = splits[city]["folds"]
            train_idx = splits[city]["train_idx"]
            test_idx = splits[city]["test_idx"]
            print(f"Tuning {label} {city}", flush=True)
            city_tune = {}
            city_tune["area_median"] = {
                "cv_mae": cv_area(frame, y, folds),
                "params": {},
                "trials": [],
            }
            city_tune["linear"] = {
                "cv_mae": cv_mae(
                    lambda: make_linear(columns, kind=kind, high=high), frame, y, folds
                ),
                "params": {},
                "trials": [],
            }
            ridge_trials = []
            for alpha in (100.0, 10.0, 1.0, 0.1):

                def _ridge(alpha=alpha):
                    return make_ridge(columns, alpha, kind=kind, high=high)

                mae = cv_mae(_ridge, frame, y, folds)
                ridge_trials.append(
                    {"mae": mae, "simplicity": [-alpha], "params": {"alpha": alpha}}
                )
            ridge_choice = choose_params(ridge_trials)
            city_tune["ridge"] = {
                "cv_mae": float(ridge_choice["mae"]),
                "params": ridge_choice["params"],
                "trials": ridge_trials,
            }
            forest_trials = []
            for depth in (8, 16):
                for leaf in (20, 5):

                    def _forest(depth=depth, leaf=leaf):
                        return make_forest(columns, depth, leaf, kind=kind, high=high)

                    mae = cv_mae(_forest, frame, y, folds)
                    forest_trials.append(
                        {
                            "mae": mae,
                            "simplicity": [depth, -leaf],
                            "params": {"max_depth": depth, "min_samples_leaf": leaf},
                        }
                    )
                    print(f"    forest depth {depth} leaf {leaf} MAE {mae:.2f}", flush=True)
            forest_choice = choose_params(forest_trials)
            city_tune["random_forest"] = {
                "cv_mae": float(forest_choice["mae"]),
                "params": forest_choice["params"],
                "trials": forest_trials,
            }
            xgb_trials = []
            for depth in (4, 6):
                for child in (20.0, 5.0):

                    def _xgb(depth=depth, child=child):
                        return make_xgb(columns, depth, child, kind=kind, high=high)

                    mae = cv_mae(_xgb, frame, y, folds)
                    xgb_trials.append(
                        {
                            "mae": mae,
                            "simplicity": [depth, -child],
                            "params": {"max_depth": depth, "min_child_weight": child},
                        }
                    )
                    print(f"    xgb depth {depth} child {child} MAE {mae:.2f}", flush=True)
            xgb_choice = choose_params(xgb_trials)
            city_tune["xgboost"] = {
                "cv_mae": float(xgb_choice["mae"]),
                "params": xgb_choice["params"],
                "trials": xgb_trials,
            }
            tuned[city] = city_tune

            print(f"Scoring {label} holdout {city}", flush=True)
            city_pred = {}
            area = AreaMedian().fit(
                frame.iloc[train_idx]["neighbourhood_cleansed"],
                frame.iloc[train_idx]["room_type"],
                y[train_idx],
            )
            area_pred = np.full(len(frame), np.nan)
            area_pred[test_idx] = area.predict(
                frame.iloc[test_idx]["neighbourhood_cleansed"],
                frame.iloc[test_idx]["room_type"],
            )
            city_pred["area_median"] = area_pred
            builders = {
                "linear": lambda: make_linear(columns, kind=kind, high=high),
                "ridge": lambda params=city_tune["ridge"]["params"]: make_ridge(
                    columns, params["alpha"], kind=kind, high=high
                ),
                "random_forest": lambda params=city_tune["random_forest"]["params"]: make_forest(
                    columns,
                    params["max_depth"],
                    params["min_samples_leaf"],
                    kind=kind,
                    high=high,
                ),
                "xgboost": lambda params=city_tune["xgboost"]["params"]: make_xgb(
                    columns,
                    params["max_depth"],
                    params["min_child_weight"],
                    kind=kind,
                    high=high,
                ),
                "neural_net": lambda: make_network(columns, kind=kind, high=high),
            }
            if label == "price":
                builders["catboost"] = lambda params=city_tune["xgboost"]["params"]: make_catboost(
                    columns, int(params["max_depth"]), kind=kind, high=high
                )
            for name, builder in builders.items():
                model = builder()
                model.fit(frame.iloc[train_idx], y[train_idx])
                pred = np.full(len(frame), np.nan)
                pred[test_idx] = model.predict(frame.iloc[test_idx])
                city_pred[name] = pred
                del model
            if label == "price":
                city_pred["blend"] = 0.5 * (city_pred["xgboost"] + city_pred["catboost"])
            predictions[city] = city_pred
        return tuned, predictions

    print("Price model comparison", flush=True)
    price_tuned, price_pred = tune_and_score("price_clean", price_columns, "log", None, "price")
    print("Nights model comparison", flush=True)
    nights_tuned, nights_pred = tune_and_score(
        "booked_nights", nights_columns, nights_kind, ANNUAL_NIGHTS_CAP, "nights"
    )

    def holdout_block(predictions, y_column):
        block = {}
        for city in STUDY_CITIES:
            y = frames[city][y_column].to_numpy(dtype=float)
            idx = splits[city]["test_idx"]
            block[city] = {
                name: _metrics_at(y, pred, idx) for name, pred in predictions[city].items()
            }
        return block

    price_holdout = holdout_block(price_pred, "price_clean")
    nights_holdout = holdout_block(nights_pred, "booked_nights")
    price_means = {name: _mean_mae(price_holdout, name) for name in price_holdout[STUDY_CITIES[0]]}
    nights_means = {
        name: _mean_mae(nights_holdout, name) for name in nights_holdout[STUDY_CITIES[0]]
    }
    price_simple = choose_simplest(
        [(name, price_means[name]) for name in ELIGIBLE_MODELS],
    )
    nights_simple = choose_simplest(
        [(name, nights_means[name]) for name in ELIGIBLE_MODELS],
    )
    shipped = choose_shipped(
        price_simple,
        price_means[price_simple],
        price_means["catboost"],
        price_means["blend"],
    )
    print(f"Price ships {shipped}. Nights ships {nights_simple}.", flush=True)

    def build_named(name, columns, params, kind, high, xgb_params):
        if name == "area_median":
            return None
        if name == "linear":
            return make_linear(columns, kind=kind, high=high)
        if name == "ridge":
            return make_ridge(columns, params["ridge"]["alpha"], kind=kind, high=high)
        if name == "random_forest":
            return make_forest(
                columns,
                params["random_forest"]["max_depth"],
                params["random_forest"]["min_samples_leaf"],
                kind=kind,
                high=high,
            )
        if name == "xgboost":
            return make_xgb(
                columns,
                params["xgboost"]["max_depth"],
                params["xgboost"]["min_child_weight"],
                kind=kind,
                high=high,
            )
        if name == "catboost":
            return make_catboost(columns, int(xgb_params["max_depth"]), kind=kind, high=high)
        if name == "blend":
            return (
                make_xgb(
                    columns,
                    xgb_params["max_depth"],
                    xgb_params["min_child_weight"],
                    kind=kind,
                    high=high,
                ),
                make_catboost(columns, int(xgb_params["max_depth"]), kind=kind, high=high),
            )
        if name == "neural_net":
            return make_network(columns, kind=kind, high=high)
        raise KeyError(name)

    def refit(name, columns, kind, high, y_column):
        models = {}
        for city in STUDY_CITIES:
            frame = frames[city]
            y = frame[y_column].to_numpy(dtype=float)
            train_idx = splits[city]["train_idx"]
            table = price_tuned if y_column == "price_clean" else nights_tuned
            params = {
                "ridge": table[city]["ridge"]["params"],
                "random_forest": table[city]["random_forest"]["params"],
                "xgboost": table[city]["xgboost"]["params"],
            }
            if name == "area_median":
                inner = AreaMedian().fit(
                    frame.iloc[train_idx]["neighbourhood_cleansed"],
                    frame.iloc[train_idx]["room_type"],
                    y[train_idx],
                )
                models[city] = _FrameArea(inner)
                continue
            built = build_named(name, columns, params, kind, high, params["xgboost"])
            if name == "blend":
                first, second = built
                first.fit(frame.iloc[train_idx], y[train_idx])
                second.fit(frame.iloc[train_idx], y[train_idx])
                models[city] = BlendModel(first, second)
            else:
                built.fit(frame.iloc[train_idx], y[train_idx])
                models[city] = built
        return models

    print("Refitting the shipped price model for the app and the interval", flush=True)
    price_models = refit(shipped, price_columns, "log", None, "price_clean")
    print("Refitting the shipped nights model", flush=True)
    nights_models = refit(
        nights_simple, nights_columns, nights_kind, ANNUAL_NIGHTS_CAP, "booked_nights"
    )

    price_intervals = {}
    price_importance = {}
    price_directions = {}
    for city in STUDY_CITIES:
        frame = frames[city]
        y = frame["price_clean"].to_numpy(dtype=float)
        train_idx = splits[city]["train_idx"]
        test_idx = splits[city]["test_idx"]
        params = price_tuned[city]

        def _builder(params=params):
            if shipped == "area_median":
                raise RuntimeError("area median interval uses its own loop")
            if shipped == "blend":
                # Interval is fit on the blend's out-of-fold average.
                return None
            return build_named(
                shipped,
                price_columns,
                _param_view(params),
                "log",
                None,
                params["xgboost"]["params"],
            )

        if shipped == "blend":

            def _blend_builder(params=params):
                def builder():
                    first = make_xgb(
                        price_columns,
                        params["xgboost"]["params"]["max_depth"],
                        params["xgboost"]["params"]["min_child_weight"],
                        kind="log",
                        high=None,
                    )
                    second = make_catboost(
                        price_columns,
                        int(params["xgboost"]["params"]["max_depth"]),
                        kind="log",
                        high=None,
                    )
                    return _PairBuilder(first, second)

                return builder

            oof = oof_predict(_blend_builder(), frame, y, splits[city]["folds"])
        elif shipped == "area_median":
            oof = _oof_area(frame, y, splits[city]["folds"])
        else:
            oof = oof_predict(_builder, frame, y, splits[city]["folds"])
        if not np.isfinite(oof[train_idx]).all():
            raise RuntimeError(f"{city} out-of-fold price predictions are incomplete")
        table = fit_interval(y[train_idx], oof[train_idx], INTERVAL_MIN_BIN)
        test_pred = price_pred[city][shipped][test_idx]
        low, high = apply_interval(test_pred, table)
        coverage = interval_metrics(y[test_idx], low, high)
        price_intervals[city] = {"table": table, "holdout": coverage}
        model = price_models[city]
        if shipped == "area_median":
            price_importance[city] = []
            price_directions[city] = []
        else:
            price_importance[city] = dollar_effects(model, frame, test_idx)
            shap_model = model.first if isinstance(model, BlendModel) else model
            try:
                shap = shap_means(shap_model, frame, test_idx)
            except Exception as exc:
                print(f"  {city} TreeSHAP skipped: {type(exc).__name__}: {exc}", flush=True)
                shap = None
            if shap:
                lookup = {row["feature"]: row for row in shap}
                for row in price_importance[city]:
                    extra = lookup.get(row["feature"], {})
                    row["mean_abs_shap"] = extra.get("mean_abs_shap")
                    row["mean_shap"] = extra.get("mean_shap")
            price_directions[city] = direction_checks(model, frame, train_idx)
        print(
            f"  {city} interval coverage {coverage['coverage']:.1%} "
            f"width ${coverage['median_width']:.0f}",
            flush=True,
        )

    print("Confirmation seeds", flush=True)
    confirm_rows = []
    price_confirm = {city: {"r2": [], "mae": [], "median_ape": []} for city in STUDY_CITIES}
    nights_confirm = {city: {"r2": [], "mae": [], "median_ape": []} for city in STUDY_CITIES}
    for seed in frozen["confirm_seeds"]:
        for city in STUDY_CITIES:
            frame = frames[city]
            split = split_frame(frame, city, int(seed))
            audit = audit_split(frame, split["train_idx"], split["test_idx"])
            if not audit["passed"]:
                raise RuntimeError(f"{city} seed {seed} failed the audit: {audit}")
            y_price = frame["price_clean"].to_numpy(dtype=float)
            y_nights = frame["booked_nights"].to_numpy(dtype=float)
            price_model = _fit_one(
                shipped,
                frames[city],
                y_price,
                split["train_idx"],
                price_columns,
                "log",
                None,
                price_tuned[city],
            )
            nights_model = _fit_one(
                nights_simple,
                frames[city],
                y_nights,
                split["train_idx"],
                nights_columns,
                nights_kind,
                ANNUAL_NIGHTS_CAP,
                nights_tuned[city],
            )
            test_idx = split["test_idx"]
            pred_price = price_model.predict(frame.iloc[test_idx])
            pred_nights = nights_model.predict(frame.iloc[test_idx])
            price_metrics = regression_metrics(y_price[test_idx], pred_price)
            nights_metrics = regression_metrics(y_nights[test_idx], pred_nights)
            for key in ("r2", "mae", "median_ape"):
                price_confirm[city][key].append(price_metrics[key])
                nights_confirm[city][key].append(nights_metrics[key])
            for index, row_index in enumerate(test_idx):
                confirm_rows.append(
                    {
                        "city": city,
                        "seed": int(seed),
                        "listing_id": int(frame.iloc[row_index]["id"]),
                        "y_price": float(y_price[row_index]),
                        "pred_price": float(pred_price[index]),
                        "y_nights": float(y_nights[row_index]),
                        "pred_nights": float(pred_nights[index]),
                    }
                )
            print(
                f"  seed {seed} {city} price R² {price_metrics['r2']:.3f} "
                f"MAE {price_metrics['mae']:.2f} nights R² {nights_metrics['r2']:.3f}",
                flush=True,
            )

    print("Bootstrap and artifacts", flush=True)
    cities_out = {}
    prediction_rows = []
    for city in STUDY_CITIES:
        frame = frames[city]
        test_idx = splits[city]["test_idx"]
        train_idx = splits[city]["train_idx"]
        y_price = frame["price_clean"].to_numpy(dtype=float)
        y_nights = frame["booked_nights"].to_numpy(dtype=float)
        price_boot = bootstrap_metrics(
            y_price[test_idx],
            price_pred[city][shipped][test_idx],
            price_pred[city]["area_median"][test_idx],
            frame["cell"].to_numpy()[test_idx],
            n_boot=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )
        nights_boot = bootstrap_metrics(
            y_nights[test_idx],
            nights_pred[city][nights_simple][test_idx],
            nights_pred[city]["area_median"][test_idx],
            frame["cell"].to_numpy()[test_idx],
            n_boot=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )
        cleared = nights_cleared(
            nights_holdout[city][nights_simple]["mae"],
            nights_holdout[city]["area_median"]["mae"],
            nights_holdout[city][nights_simple]["r2"],
        )
        revenue_y = y_price[test_idx] * y_nights[test_idx]
        revenue_p = price_pred[city][shipped][test_idx] * nights_pred[city][nights_simple][test_idx]
        revenue = regression_metrics(revenue_y, revenue_p)
        train_revenue = y_price[train_idx] * y_nights[train_idx]
        cuts = {
            f"p{pct}": float(np.quantile(train_revenue, pct / 100.0)) for pct in (20, 40, 60, 80)
        }
        match = [
            grade_revenue(float(pred), cuts)["letter"]
            == grade_revenue(float(actual), cuts)["letter"]
            for pred, actual in zip(revenue_p, revenue_y, strict=True)
        ]
        part = frame.iloc[train_idx]
        grouped = part.groupby("neighbourhood_cleansed")["booked_nights"].agg(["median", "count"])
        by_neighbourhood = {
            str(key): float(value)
            for key, value in grouped.loc[grouped["count"] >= 8, "median"].items()
        }
        low, high = apply_interval(
            price_pred[city][shipped][test_idx], price_intervals[city]["table"]
        )
        for position, row_index in enumerate(test_idx):
            row = {
                "city": city,
                "listing_id": int(frame.iloc[row_index]["id"]),
                "cell": int(frame.iloc[row_index]["cell"]),
                "y_price": float(y_price[row_index]),
                "y_nights": float(y_nights[row_index]),
                "pred_lo": float(low[position]),
                "pred_hi": float(high[position]),
            }
            for name, pred in price_pred[city].items():
                row[f"price_{name}"] = float(pred[row_index])
            for name, pred in nights_pred[city].items():
                row[f"nights_{name}"] = float(pred[row_index])
            prediction_rows.append(row)

        def _confirm_block(store, primary_metrics):
            return {
                "seeds": [int(seed) for seed in frozen["confirm_seeds"]],
                "r2": store["r2"],
                "mae": store["mae"],
                "median_ape": store["median_ape"],
                "primary_r2": primary_metrics["r2"],
                "primary_mae": primary_metrics["mae"],
                "r2_min": float(min(store["r2"])),
                "r2_max": float(max(store["r2"])),
                "r2_spread": float(max(store["r2"]) - min(store["r2"])),
                "mae_min": float(min(store["mae"])),
                "mae_max": float(max(store["mae"])),
                "mae_spread": float(max(store["mae"]) - min(store["mae"])),
            }

        info = splits[city]["info"]
        cities_out[city] = {
            "display": CITIES[city]["display"],
            "n_raw": reports[city]["rows_raw"],
            "n_clean": reports[city]["rows_final"],
            "n_train": int(info["train_after"]),
            "n_test": int(info["n_test"]),
            "removed_host": int(info["removed_host"]),
            "removed_buffer": int(info["removed_buffer"]),
            "audit": splits[city]["audit"],
            "n_folds": splits[city]["n_splits"],
            "price": {
                "n_test_cells": int(info["n_test_cells"]),
                "n_train_cells": int(info["n_train_cells"]),
                "holdout": price_holdout[city],
                "cv_mae": {name: price_tuned[city][name]["cv_mae"] for name in ELIGIBLE_MODELS},
                "params": {
                    name: price_tuned[city][name]["params"]
                    for name in ("ridge", "random_forest", "xgboost")
                },
                "trials": {
                    name: price_tuned[city][name]["trials"]
                    for name in ("ridge", "random_forest", "xgboost")
                },
                "bootstrap": price_boot,
                "interval": price_intervals[city]["holdout"],
                "importance": price_importance[city],
                "directions": price_directions[city],
                "confirmation": _confirm_block(price_confirm[city], price_holdout[city][shipped]),
            },
            "nights": {
                "holdout": nights_holdout[city],
                "cv_mae": {name: nights_tuned[city][name]["cv_mae"] for name in ELIGIBLE_MODELS},
                "params": {
                    name: nights_tuned[city][name]["params"]
                    for name in ("ridge", "random_forest", "xgboost")
                },
                "trials": {
                    name: nights_tuned[city][name]["trials"]
                    for name in ("ridge", "random_forest", "xgboost")
                },
                "bootstrap": nights_boot,
                "cleared_bar": bool(cleared),
                "confirmation": _confirm_block(
                    nights_confirm[city], nights_holdout[city][nights_simple]
                ),
                "revenue": {"r2": revenue["r2"], "mae": revenue["mae"]},
                "grade_cuts": cuts,
                "grade_match": float(np.mean(match)),
            },
        }

        artifact = {
            "kind": "simple-study",
            "city": city,
            "library": shipped,
            "model": price_models[city],
            "columns": list(price_columns),
            "medians": dict(getattr(price_models[city], "medians_", {})),
            "interval": price_intervals[city]["table"],
            "interval_coverage": price_intervals[city]["holdout"]["coverage"],
            "interval_median_width": price_intervals[city]["holdout"]["median_width"],
            "price_holdout_r2": price_holdout[city][shipped]["r2"],
            "price_holdout_mae": price_holdout[city][shipped]["mae"],
            "price_baseline_mae": price_holdout[city]["area_median"]["mae"],
            "nights_library": nights_simple,
            "nights_model": nights_models[city],
            "nights_columns": list(nights_columns),
            "nights_medians": dict(getattr(nights_models[city], "medians_", {})),
            "nights_city_median": float(part["booked_nights"].median()),
            "nights_by_neighbourhood": by_neighbourhood,
            "grade_cuts": cuts,
        }
        save_artifact(city, artifact)
        print(f"  saved {city}", flush=True)

    every_city_cleared = all(cities_out[city]["nights"]["cleared_bar"] for city in STUDY_CITIES)
    # The rough-estimate flag is global: one city missing the bar labels every city.
    for city in STUDY_CITIES:
        artifact = __import__("models.artifacts", fromlist=["load_artifact"]).load_artifact(city)
        artifact["nights_rough_estimate"] = not every_city_cleared
        artifact["nights_holdout_r2"] = nights_holdout[city][nights_simple]["r2"]
        artifact["nights_holdout_mae"] = nights_holdout[city][nights_simple]["mae"]
        artifact["nights_baseline_mae"] = nights_holdout[city]["area_median"]["mae"]
        artifact["nights_cleared_bar"] = bool(cities_out[city]["nights"]["cleared_bar"])
        save_artifact(city, artifact)

    simple_scores = [(name, price_means[name]) for name in ELIGIBLE_MODELS]
    payload = {
        "study": "simple-spatial",
        "primary_seed": int(frozen["primary_seed"]),
        "confirm_seeds": [int(seed) for seed in frozen["confirm_seeds"]],
        "cell_km": 3.0,
        "holdout_fraction": 0.20,
        "buffer_km": 1.0,
        "max_km_from_center": 60.0,
        "bootstrap_draws": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "price_features": price_columns,
        "price_feature_name": price_feature_name,
        "price_feature_cv": {
            "mean": price_feature_mean,
            "by_city": price_feature_by_city,
        },
        "price_dropped_sparse": price_dropped,
        "nights_features": nights_columns,
        "nights_feature_name": nights_feature_name,
        "nights_feature_cv": {
            "mean": nights_feature_mean,
            "by_city": nights_feature_by_city,
        },
        "nights_dropped_sparse": nights_dropped,
        "nights_target": nights_target,
        "nights_target_cv": {
            "raw": raw_mae,
            "log": log_mae,
            "by_city_raw": raw_by_city,
            "by_city_log": log_by_city,
        },
        "nights_target_sentence": nights_target_sentence,
        "price_mean_mae": price_means,
        "nights_mean_mae": nights_means,
        "simple_model": price_simple,
        "shipped_model": shipped,
        "selection_reason": selection_reason(simple_scores, price_simple),
        "challenger_reason": challenger_reason(
            price_simple,
            price_means[price_simple],
            price_means["catboost"],
            price_means["blend"],
        ),
        "catboost_shipped": shipped == "catboost",
        "blend_shipped": shipped == "blend",
        "nights_shipped_model": nights_simple,
        "nights_selection_reason": nights_selection_reason(
            [(name, nights_means[name]) for name in ELIGIBLE_MODELS],
            nights_simple,
        ),
        "nights_cleared_bar": bool(every_city_cleared),
        "nights_rough_estimate": not every_city_cleared,
        "nights_reason": nights_reason(every_city_cleared),
        "audit_passed": all(cities_out[city]["audit"]["passed"] for city in STUDY_CITIES),
        "cities": cities_out,
    }
    if not payload["audit_passed"]:
        raise RuntimeError("audit failed")
    _write(REPORTS / "study_metrics.json", payload)
    pd.DataFrame(prediction_rows).to_csv(REPORTS / "study_predictions.csv", index=False)
    pd.DataFrame(confirm_rows).to_csv(REPORTS / "study_confirmation.csv", index=False)
    print(payload["selection_reason"], flush=True)
    print(payload["challenger_reason"], flush=True)
    print(payload["nights_reason"], flush=True)
    print("Wrote reports/study_metrics.json", flush=True)
    return payload


class _FrameArea:
    """Area median with a DataFrame predict(), for the app and confirmation."""

    def __init__(self, inner):
        self.inner = inner
        self.columns: list[str] = []
        self.medians_: dict[str, float] = {}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.inner.predict(frame["neighbourhood_cleansed"], frame["room_type"])


class _PairBuilder:
    """Out-of-fold blend. fit/predict match LogModel so oof_predict can call them."""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def fit(self, frame, y):
        self.first.fit(frame, y)
        self.second.fit(frame, y)
        return self

    def predict(self, frame):
        return 0.5 * (self.first.predict(frame) + self.second.predict(frame))


def _param_view(params: dict) -> dict:
    return {
        "ridge": params["ridge"]["params"],
        "random_forest": params["random_forest"]["params"],
        "xgboost": params["xgboost"]["params"],
    }


def _oof_area(frame, y, folds) -> np.ndarray:
    from evaluation.study_models import AreaMedian

    pred = np.full(len(frame), np.nan)
    for train_idx, valid_idx in folds:
        model = AreaMedian().fit(
            frame.iloc[train_idx]["neighbourhood_cleansed"],
            frame.iloc[train_idx]["room_type"],
            y[train_idx],
        )
        pred[valid_idx] = model.predict(
            frame.iloc[valid_idx]["neighbourhood_cleansed"],
            frame.iloc[valid_idx]["room_type"],
        )
    return pred


def _fit_one(name, frame, y, train_idx, columns, kind, high, tuned):
    """Fit one city on one split. Used for the confirmation seeds."""
    from evaluation.study_models import (
        AreaMedian,
        BlendModel,
        make_catboost,
        make_forest,
        make_linear,
        make_network,
        make_ridge,
        make_xgb,
    )

    if name == "area_median":
        inner = AreaMedian().fit(
            frame.iloc[train_idx]["neighbourhood_cleansed"],
            frame.iloc[train_idx]["room_type"],
            y[train_idx],
        )
        return _FrameArea(inner)
    if name == "linear":
        model = make_linear(columns, kind=kind, high=high)
    elif name == "ridge":
        model = make_ridge(columns, tuned["ridge"]["params"]["alpha"], kind=kind, high=high)
    elif name == "random_forest":
        model = make_forest(
            columns,
            tuned["random_forest"]["params"]["max_depth"],
            tuned["random_forest"]["params"]["min_samples_leaf"],
            kind=kind,
            high=high,
        )
    elif name == "xgboost":
        model = make_xgb(
            columns,
            tuned["xgboost"]["params"]["max_depth"],
            tuned["xgboost"]["params"]["min_child_weight"],
            kind=kind,
            high=high,
        )
    elif name == "catboost":
        model = make_catboost(
            columns, int(tuned["xgboost"]["params"]["max_depth"]), kind=kind, high=high
        )
    elif name == "blend":
        first = make_xgb(
            columns,
            tuned["xgboost"]["params"]["max_depth"],
            tuned["xgboost"]["params"]["min_child_weight"],
            kind=kind,
            high=high,
        )
        second = make_catboost(
            columns, int(tuned["xgboost"]["params"]["max_depth"]), kind=kind, high=high
        )
        first.fit(frame.iloc[train_idx], y[train_idx])
        second.fit(frame.iloc[train_idx], y[train_idx])
        return BlendModel(first, second)
    elif name == "neural_net":
        model = make_network(columns, kind=kind, high=high)
    else:
        raise KeyError(name)
    model.fit(frame.iloc[train_idx], y[train_idx])
    return model


def main() -> None:
    run()


if __name__ == "__main__":
    main()
