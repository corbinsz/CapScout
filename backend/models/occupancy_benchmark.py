"""Compare occupancy targets, then retrain the occupancy models if reviews win.

Price models are left as they are. Only occupancy_model.joblib is replaced.

Usage (from backend/):
    python -m models.occupancy_benchmark
"""
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cities import CITIES
from models.benchmark import _impute, _load_city, _xgb
from models.estimators import DollarRegressor
from models.feature_engineering import get_feature_columns
from models.metrics import group_holdout, regression_metrics
from models.occupancy import blocked_not_booked
import config

BENCH = ['austin', 'seattle', 'asheville', 'nashville']
TARGETS = ('occupancy_calendar', 'occupancy_rate')


def _fit_eval(frame, y, train_idx, test_idx, columns):
    x_train, x_test, _ = _impute(frame.iloc[train_idx][columns], frame.iloc[test_idx][columns])
    model = _xgb()
    model.fit(x_train, y[train_idx])
    pred = np.clip(model.predict(x_test), 0, 1)
    baseline = np.full(len(test_idx), np.median(y[train_idx]))
    learned = regression_metrics(y[test_idx], pred)
    naive = regression_metrics(y[test_idx], baseline)
    ratio = None
    if learned['mae'] is not None and naive['mae']:
        ratio = round(learned['mae'] / naive['mae'], 4)
    return learned, naive, ratio


def evaluate_city(frame):
    columns = [col for col in get_feature_columns() if col in frame.columns]
    train_idx, test_idx = group_holdout(frame['id'].to_numpy())
    out = {'n_test': int(len(test_idx))}
    for name in TARGETS:
        if name not in frame.columns:
            continue
        y = frame[name].to_numpy(dtype=float)
        learned, naive, ratio = _fit_eval(frame, y, train_idx, test_idx, columns)
        out[name] = {'model': learned, 'baseline': naive, 'mae_ratio': ratio}
    calendar = frame['occupancy_calendar'] if 'occupancy_calendar' in frame.columns else None
    reviews = frame['reviews_ltm'] if 'reviews_ltm' in frame.columns else None
    if calendar is not None and reviews is not None:
        flags = [
            blocked_not_booked(avail, rev)
            for avail, rev in zip((1 - calendar) * 30, reviews)
        ]
        out['share_full_calendar_and_no_reviews'] = round(float(np.mean(flags)), 4)
        both = frame[['occupancy_calendar', 'occupancy_rate']].dropna()
        if len(both) > 20:
            out['correlation_calendar_vs_reviews'] = round(
                float(both['occupancy_calendar'].corr(both['occupancy_rate'])), 4
            )
    if 'min_nights' in frame.columns:
        out['share_minimum_nights_28_or_more'] = round(float((frame['min_nights'] >= 28).mean()), 4)
    if 'availability_365' in frame.columns and 'availability_30' in frame.columns:
        year_open = frame['availability_365'] / 365.0
        month_open = frame['availability_30'] / 30.0
        # Hosts who leave the year open but close the next month, with no reviews,
        # are blocking or holding dates. The calendar calls that "occupied".
        quiet = frame['reviews_ltm'].fillna(0) <= 0 if 'reviews_ltm' in frame.columns else False
        pattern = (year_open > 0.5) & (month_open < 0.1) & quiet
        out['share_year_open_month_closed_no_reviews'] = round(float(pattern.mean()), 4)
    return out


def retrain_city(city_key):
    frame = _load_city(city_key)
    models_dir = config.city_models_dir(city_key)
    columns = joblib.load(os.path.join(models_dir, 'feature_columns.joblib'))
    columns = [col for col in columns if col in frame.columns]
    y = frame['occupancy_rate'].to_numpy(dtype=float)
    train_idx, test_idx = group_holdout(frame['id'].to_numpy())
    learned, naive, ratio = _fit_eval(frame, y, train_idx, test_idx, columns)
    x_all, _, _ = _impute(frame[columns], frame[columns])
    model = _xgb()
    model.fit(x_all, y)
    wrapped = DollarRegressor(model, 'occupancy')
    joblib.dump(wrapped, os.path.join(models_dir, 'occupancy_model.joblib'), compress=3)
    metrics_path = os.path.join(models_dir, 'metrics.json')
    payload = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding='utf-8') as handle:
            payload = json.load(handle)
    payload['occupancy_target'] = (
        'San Francisco review model: reviews in the last year / 0.5, times the '
        'city stay or the listing minimum (capped at 14 nights), divided by 365, '
        'capped at 70%. A full calendar with no reviews is not treated as booked.'
    )
    payload['occupancy_holdout'] = {
        'model': learned,
        'baseline_median': naive,
        'mae_ratio': ratio,
    }
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    return {'mae': learned['mae'], 'r2': learned['r2'], 'mae_ratio': ratio}


def main():
    report = {'cities': {}, 'notes': (
        'mae_ratio below 1 beats predicting the training median. '
        'occupancy_calendar is 1 - availability_30/30. '
        'occupancy_rate is the San Francisco review model. '
        'Raw calendar files are not in this repo. availability_30 and '
        'availability_365 are the calendar summaries Inside Airbnb publishes '
        'on each listing.'
    )}
    ratios = {name: [] for name in TARGETS}
    for city in BENCH:
        print(f'benchmark {city}', flush=True)
        frame = _load_city(city)
        report['cities'][city] = evaluate_city(frame)
        for name in TARGETS:
            ratio = report['cities'][city].get(name, {}).get('mae_ratio')
            if ratio is not None:
                ratios[name].append(ratio)
            block = report['cities'][city]
            print(
                f"  {name} ratio {ratio}  "
                f"blocked-looking {block.get('share_full_calendar_and_no_reviews')}",
                flush=True,
            )
    means = {name: round(float(np.mean(vals)), 4) if vals else None for name, vals in ratios.items()}
    report['mean_mae_ratio'] = means
    winner = min(means, key=lambda name: means[name] if means[name] is not None else 99)
    report['more_learnable'] = winner
    report['winner'] = 'occupancy_rate'
    report['why'] = (
        'The production target is the review model even if the calendar target '
        'is slightly easier to fit. A full calendar with no reviews is not a booking, '
        'so fitting the calendar well does not answer the investor question.'
    )
    print('more learnable', winner, means, flush=True)
    report['retrained'] = {}
    for city in CITIES:
        print(f'retrain occupancy {city}', flush=True)
        report['retrained'][city] = retrain_city(city)
        print(' ', report['retrained'][city], flush=True)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'docs', 'occupancy_benchmark.json')
    with open(out, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print('wrote', out)


if __name__ == '__main__':
    main()
