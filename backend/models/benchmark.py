"""Compare listing models on a real holdout, then keep the winner.

The previous production models were fit on every row after each row's price had
been replaced by an area median that included that same row. Cross-validated
R-squared on that target landed near 0.97 and did not describe error on a
home the model had not seen.

This script:
  1. Scores the saved models on a 20% listing holdout (the "before" numbers).
  2. Fits ridge, elastic net, histogram gradient boosting, XGBoost, XGBoost with
     extra spatial features, and XGBoost trained on train-only area medians.
  3. Picks the lowest combined error on the benchmark cities.
  4. Retrains that approach for every city and writes metrics.json beside it.

Usage (from backend/):
    python -m models.benchmark
    python -m models.benchmark --cities austin,asheville
"""
import argparse
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from cities import CITIES
from models.estimators import DollarRegressor
from models.feature_engineering import get_feature_columns, load_and_clean_listings, smooth_targets
from models.metrics import group_holdout, regression_metrics, time_holdout

warnings.filterwarnings('ignore', category=UserWarning)

BENCH_CITIES = ['austin', 'seattle', 'asheville', 'nashville']
SPATIAL_FEATURES = ['dist_sq', 'lat_x_lng', 'density_3km']
APPROACHES = [
    'ridge',
    'elasticnet',
    'hist_gb',
    'xgboost',
    'xgboost_spatial',
    'xgboost_area_median',
]
EARTH_KM = 6371.0


def _xgb():
    return XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        tree_method='hist',
        n_jobs=4,
        random_state=42,
        verbosity=0,
    )


def _make_estimator(approach):
    if approach == 'ridge':
        return Pipeline([
            ('scale', StandardScaler()),
            ('model', Ridge(alpha=3.0)),
        ])
    if approach == 'elasticnet':
        return Pipeline([
            ('scale', StandardScaler()),
            ('model', ElasticNet(alpha=0.005, l1_ratio=0.4, max_iter=8000)),
        ])
    if approach == 'hist_gb':
        return HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.08, random_state=42,
        )
    return _xgb()


def _uses_spatial(approach):
    return approach == 'xgboost_spatial'


def _uses_area_target(approach):
    return approach == 'xgboost_area_median'


def _prepare_frame(df):
    frame = df.copy()
    frame['dist_sq'] = frame['distance_to_center'] ** 2
    frame['lat_x_lng'] = frame['latitude'] * frame['longitude']
    frame['density_3km'] = 0.0
    return frame


def _fill_density(frame, train_idx):
    """Nearby-listing counts from the training rows only."""
    train = frame.iloc[train_idx]
    coords = np.deg2rad(np.c_[train['latitude'].to_numpy(), train['longitude'].to_numpy()])
    if len(coords) < 5:
        return frame
    tree = BallTree(coords, metric='haversine')
    all_coords = np.deg2rad(np.c_[frame['latitude'].to_numpy(), frame['longitude'].to_numpy()])
    counts = tree.query_radius(all_coords, r=3.0 / EARTH_KM, count_only=True)
    frame = frame.copy()
    frame['density_3km'] = counts.astype(float)
    return frame


def _impute(train_x, other_x):
    medians = train_x.median(numeric_only=True)
    return train_x.fillna(medians).fillna(0), other_x.fillna(medians).fillna(0), medians.to_dict()


def _fit_predict(approach, x_train, y_train, x_test, kind):
    model = _make_estimator(approach)
    if kind == 'price':
        model.fit(x_train, np.log1p(np.clip(y_train, 0, None)))
        pred = np.expm1(model.predict(x_test))
        pred = np.clip(pred, 15, 5000)
    else:
        model.fit(x_train, np.clip(y_train, 0, 1))
        pred = np.clip(model.predict(x_test), 0, 1)
    return model, pred


def _area_targets(frame, price, occ, train_idx):
    mask = np.zeros(len(frame), dtype=bool)
    mask[train_idx] = True
    columns = [col for col in ['latitude', 'longitude', 'bedrooms'] if col in frame.columns]
    smoothed_price, smoothed_occ, _, _ = smooth_targets(
        frame[columns],
        np.asarray(price, dtype=float),
        np.asarray(occ, dtype=float),
        frame['id'].to_numpy(),
        train_mask=mask,
    )
    return smoothed_price, smoothed_occ


def _score_bundle(y_price, pred_price, y_price_area, y_occ, pred_occ, baseline_price, baseline_occ, baseline_area):
    price = regression_metrics(y_price, pred_price)
    area = regression_metrics(y_price_area, pred_price)
    occ = regression_metrics(y_occ, pred_occ)
    base_price = regression_metrics(y_price, baseline_price)
    base_area = regression_metrics(y_price_area, baseline_area)
    base_occ = regression_metrics(y_occ, baseline_occ)

    def ratio(metric, base):
        if not metric['mae'] or not base['mae']:
            return 1.0
        return metric['mae'] / base['mae']

    return {
        'price_raw': price,
        'price_area': area,
        'occupancy_raw': occ,
        'baseline_price_mae': base_price['mae'],
        'baseline_area_mae': base_area['mae'],
        'baseline_occ_mae': base_occ['mae'],
        'ratios': {
            'area': ratio(area, base_area),
            'raw': ratio(price, base_price),
            'occ': ratio(occ, base_occ),
        },
    }


def _combined_score(holdout, time_split):
    """Lower is better. 1.0 means 'no better than predicting the training median'.

    Raw nightly price and the time split dominate. Area-median error is reported
    in the tables but is not allowed to choose the model: a model trained on
    area medians wins that column by construction and can be worse on real prices.
    """
    ratios = holdout['ratios']
    parts = [0.55 * ratios['raw'], 0.25 * ratios['occ']]
    weight = 0.80
    if time_split and time_split.get('ratios'):
        parts.append(0.20 * time_split['ratios']['raw'])
        weight = 1.0
    return round(sum(parts) / weight, 4)


def evaluate_approach(frame, approach):
    price = frame['price_clean'].to_numpy(dtype=float)
    occ = frame['occupancy_rate'].to_numpy(dtype=float)
    train_idx, test_idx = group_holdout(frame['id'].to_numpy())
    working = _fill_density(frame, train_idx) if _uses_spatial(approach) else frame
    columns = get_feature_columns() + (SPATIAL_FEATURES if _uses_spatial(approach) else [])
    columns = [col for col in columns if col in working.columns]
    area_price, area_occ = _area_targets(working, price, occ, train_idx)
    y_price = area_price if _uses_area_target(approach) else price
    y_occ = area_occ if _uses_area_target(approach) else occ

    x_train, x_test, _medians = _impute(working.iloc[train_idx][columns], working.iloc[test_idx][columns])
    _, pred_price = _fit_predict(approach, x_train, y_price[train_idx], x_test, 'price')
    _, pred_occ = _fit_predict(approach, x_train, y_occ[train_idx], x_test, 'occupancy')
    baseline_price = np.full(len(test_idx), np.median(price[train_idx]))
    baseline_area = np.full(len(test_idx), np.median(area_price[train_idx]))
    baseline_occ = np.full(len(test_idx), np.median(occ[train_idx]))
    holdout = _score_bundle(
        price[test_idx], pred_price, area_price[test_idx],
        occ[test_idx], pred_occ, baseline_price, baseline_occ, baseline_area,
    )

    time_split = None
    stamps = pd.to_datetime(working['last_review'], errors='coerce') if 'last_review' in working.columns else None
    if stamps is not None:
        split = time_holdout(stamps.to_numpy())
        if split is not None:
            tr, te = split
            timed = _fill_density(frame, tr) if _uses_spatial(approach) else frame
            area_p, area_o = _area_targets(timed, price, occ, tr)
            y_p = area_p if _uses_area_target(approach) else price
            y_o = area_o if _uses_area_target(approach) else occ
            xt, xe, _ = _impute(timed.iloc[tr][columns], timed.iloc[te][columns])
            _, p_price = _fit_predict(approach, xt, y_p[tr], xe, 'price')
            _, p_occ = _fit_predict(approach, xt, y_o[tr], xe, 'occupancy')
            time_split = _score_bundle(
                price[te], p_price, area_p[te], occ[te], p_occ,
                np.full(len(te), np.median(price[tr])),
                np.full(len(te), np.median(occ[tr])),
                np.full(len(te), np.median(area_p[tr])),
            )
            time_split['n_test'] = int(len(te))
    result = {
        'holdout': holdout,
        'time_split': time_split,
        'score': _combined_score(holdout, time_split),
        'n_train': int(len(train_idx)),
        'n_test': int(len(test_idx)),
    }
    return result


def _saved_feature_frame(frame, city_key):
    models_dir = config.city_models_dir(city_key)
    price_path = os.path.join(models_dir, 'price_model.joblib')
    occ_path = os.path.join(models_dir, 'occupancy_model.joblib')
    if not (os.path.isfile(price_path) and os.path.isfile(occ_path)):
        return None
    monthly_cols_path = os.path.join(models_dir, 'monthly_feature_columns.joblib')
    feature_cols = joblib.load(os.path.join(models_dir, 'feature_columns.joblib'))
    medians = joblib.load(os.path.join(models_dir, 'feature_medians.joblib'))
    price_model = joblib.load(price_path)
    occ_model = joblib.load(occ_path)
    base = frame.copy()
    for col in feature_cols:
        if col not in base.columns:
            base[col] = np.nan
    base[feature_cols] = base[feature_cols].fillna(pd.Series(medians))
    base[feature_cols] = base[feature_cols].fillna(0)
    if os.path.isfile(monthly_cols_path):
        monthly_cols = joblib.load(monthly_cols_path)
        monthly_medians_path = os.path.join(models_dir, 'monthly_feature_medians.joblib')
        monthly_medians = joblib.load(monthly_medians_path) if os.path.isfile(monthly_medians_path) else medians
        chunks = []
        for month in range(1, 13):
            chunk = base[feature_cols].copy()
            chunk['sin_month'] = np.sin(2 * np.pi * month / 12)
            chunk['cos_month'] = np.cos(2 * np.pi * month / 12)
            chunks.append(chunk)
        expanded = pd.concat(chunks, ignore_index=True)
        for col in monthly_cols:
            if col not in expanded.columns:
                expanded[col] = monthly_medians.get(col, 0)
        expanded = expanded[monthly_cols]
        rates = np.asarray(price_model.predict(expanded), dtype=float).reshape(len(frame), 12).mean(axis=1)
        occs = np.clip(np.asarray(occ_model.predict(expanded), dtype=float), 0, 1).reshape(len(frame), 12).mean(axis=1)
    else:
        rates = np.asarray(price_model.predict(base[feature_cols]), dtype=float)
        occs = np.clip(np.asarray(occ_model.predict(base[feature_cols]), dtype=float), 0, 1)
    return rates, occs


def evaluate_saved(frame, city_key):
    predicted = _saved_feature_frame(frame, city_key)
    if predicted is None:
        return None
    rates, occs = predicted
    price = frame['price_clean'].to_numpy(dtype=float)
    occ = frame['occupancy_rate'].to_numpy(dtype=float)
    train_idx, test_idx = group_holdout(frame['id'].to_numpy())
    area_price, _area_occ = _area_targets(frame, price, occ, train_idx)
    baseline_price = np.full(len(test_idx), np.median(price[train_idx]))
    baseline_area = np.full(len(test_idx), np.median(area_price[train_idx]))
    baseline_occ = np.full(len(test_idx), np.median(occ[train_idx]))
    holdout = _score_bundle(
        price[test_idx], rates[test_idx], area_price[test_idx],
        occ[test_idx], occs[test_idx], baseline_price, baseline_occ, baseline_area,
    )
    return {
        'holdout': holdout,
        'score': _combined_score(holdout, None),
        'n_test': int(len(test_idx)),
        'note': (
            'Saved model scored on listings it was not asked to hold out. '
            'It was originally fit on all rows, so this number is slightly '
            'optimistic versus a model that never saw these homes.'
        ),
    }


def _feature_stats(frame, columns, price, occ):
    stats = {}
    for col in columns:
        series = frame[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            'min': float(series.min()),
            'max': float(series.max()),
            'p5': float(series.quantile(0.05)),
            'p25': float(series.quantile(0.25)),
            'p50': float(series.quantile(0.50)),
            'p75': float(series.quantile(0.75)),
            'p95': float(series.quantile(0.95)),
            'mean': float(series.mean()),
            'std': float(series.std() or 0),
        }
    return {
        'features': stats,
        'targets': {
            'price_mean': float(np.mean(price)),
            'price_std': float(np.std(price)),
            'occ_mean': float(np.mean(occ)),
            'occ_std': float(np.std(occ)),
            'n_training': int(len(frame)),
        },
        'training_coords': {
            'latitude': frame['latitude'].to_numpy(dtype=np.float32),
            'longitude': frame['longitude'].to_numpy(dtype=np.float32),
        },
    }


def train_city(frame, city_key, approach, before):
    """Fit the chosen approach on a holdout, record metrics, refit on every row."""
    evaluated = evaluate_approach(frame, approach)
    price = frame['price_clean'].to_numpy(dtype=float)
    occ = frame['occupancy_rate'].to_numpy(dtype=float)
    all_idx = np.arange(len(frame))
    working = _fill_density(frame, all_idx) if _uses_spatial(approach) else frame
    columns = get_feature_columns() + (SPATIAL_FEATURES if _uses_spatial(approach) else [])
    columns = [col for col in columns if col in working.columns]
    if _uses_area_target(approach):
        y_price, y_occ = _area_targets(working, price, occ, all_idx)
    else:
        y_price, y_occ = price, occ
    x_all, _, medians = _impute(working[columns], working[columns])
    price_model, _ = _fit_predict(approach, x_all, y_price, x_all.iloc[:1], 'price')
    occ_model, _ = _fit_predict(approach, x_all, y_occ, x_all.iloc[:1], 'occupancy')
    price_model = DollarRegressor(price_model, 'price')
    occ_model = DollarRegressor(occ_model, 'occupancy')

    models_dir = config.city_models_dir(city_key)
    joblib.dump(price_model, os.path.join(models_dir, 'price_model.joblib'), compress=3)
    joblib.dump(occ_model, os.path.join(models_dir, 'occupancy_model.joblib'), compress=3)
    joblib.dump(columns, os.path.join(models_dir, 'feature_columns.joblib'), compress=3)
    joblib.dump(medians, os.path.join(models_dir, 'feature_medians.joblib'), compress=3)
    joblib.dump(_feature_stats(working, columns, price, occ), os.path.join(models_dir, 'feature_stats.joblib'), compress=3)
    for stale in ('monthly_feature_columns.joblib', 'monthly_feature_medians.joblib'):
        path = os.path.join(models_dir, stale)
        if os.path.isfile(path):
            os.remove(path)

    payload = {
        'approach': approach,
        'target': 'train-only area median' if _uses_area_target(approach) else 'raw listing price and occupancy',
        'features': columns,
        'price_transform': 'log1p',
        'seasonality': 'Applied at prediction time from city seasonal factors. The model itself predicts the annual typical rate.',
        'occupancy_target': '1 - availability_30/30. Host blocks look like bookings, so occupancy error stays large.',
        'holdout': evaluated['holdout'],
        'time_split': evaluated['time_split'],
        'score': evaluated['score'],
        'n_train': evaluated['n_train'],
        'n_test': evaluated['n_test'],
        'before': before,
    }
    with open(os.path.join(models_dir, 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _load_city(city_key):
    df = load_and_clean_listings(city_key)
    needed = ['id', 'price_clean', 'occupancy_rate', 'latitude', 'longitude', 'bedrooms']
    df = df.dropna(subset=needed)
    df = df[df['price_clean'] > 0]
    if 'last_review' not in df.columns:
        df['last_review'] = pd.NaT
    return _prepare_frame(df.reset_index(drop=True))


def run(cities, bench_cities, apply):
    started = time.time()
    before = {}
    frames = {}
    for city_key in cities:
        print(f'\n=== before: {city_key} ===', flush=True)
        frame = _load_city(city_key)
        frames[city_key] = frame
        print(f'  {len(frame)} listings', flush=True)
        before[city_key] = evaluate_saved(frame, city_key)
        if before[city_key]:
            holdout = before[city_key]['holdout']
            print(
                f"  saved model  raw MAE ${holdout['price_raw']['mae']:.0f}  "
                f"area MAE ${holdout['price_area']['mae']:.0f}  "
                f"occ MAE {holdout['occupancy_raw']['mae']:.3f}",
                flush=True,
            )

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'docs')
    os.makedirs(results_dir, exist_ok=True)
    partial_path = os.path.join(results_dir, 'ml_benchmark.json')

    comparison = {approach: {} for approach in APPROACHES}
    for city_key in bench_cities:
        frame = frames[city_key]
        for approach in APPROACHES:
            t0 = time.time()
            print(f'  fit {approach} on {city_key}...', flush=True)
            comparison[approach][city_key] = evaluate_approach(frame, approach)
            print(f'    score {comparison[approach][city_key]["score"]} in {time.time() - t0:.1f}s', flush=True)

    means = {}
    for approach, by_city in comparison.items():
        vals = [payload['score'] for payload in by_city.values()]
        means[approach] = round(float(np.mean(vals)), 4) if vals else None
    winner = min(means, key=lambda name: means[name] if means[name] is not None else 99)
    print(f'\nWinner: {winner}  mean score {means[winner]}', flush=True)
    print('Lower is better. 1.0 matches predicting the training median.', flush=True)

    after = {}
    if apply:
        for city_key in cities:
            print(f'\n=== train {winner}: {city_key} ===', flush=True)
            t0 = time.time()
            after[city_key] = train_city(frames[city_key], city_key, winner, before.get(city_key))
            print(f'  saved in {time.time() - t0:.1f}s  score {after[city_key]["score"]}', flush=True)

    report = {
        'winner': winner,
        'mean_score_by_approach': means,
        'score_definition': (
            'Ratio of MAE to a training-median baseline. '
            '0.55 raw nightly price on a listing holdout, 0.25 raw occupancy, '
            '0.20 time-split raw nightly price when last_review dates exist. '
            'Area-median error is reported and is not part of the selection score, '
            'because training on area medians wins that column automatically. '
            'Below 1 beats predicting the median.'
        ),
        'split': 'GroupShuffleSplit by listing id, 20% test, seed 42. Time split keeps the newest 20% of listings that have a last_review date.',
        'benchmark_cities': bench_cities,
        'comparison': comparison,
        'before': before,
        'after': {city: {
            'score': payload['score'],
            'holdout': payload['holdout'],
            'time_split': payload['time_split'],
        } for city, payload in after.items()},
        'seconds': round(time.time() - started, 1),
    }
    with open(partial_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nWrote {partial_path}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', default=','.join(CITIES.keys()))
    parser.add_argument('--bench-cities', default=','.join(BENCH_CITIES))
    parser.add_argument('--no-apply', action='store_true', help='Compare only; do not replace saved models')
    args = parser.parse_args()
    cities = [city for city in args.cities.split(',') if city]
    bench = [city for city in args.bench_cities.split(',') if city]
    run(cities, bench, apply=not args.no_apply)


if __name__ == '__main__':
    main()
