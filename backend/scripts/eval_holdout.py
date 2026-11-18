"""Listing-grouped holdout for the price and occupancy models.

This does not load or overwrite the saved models in models/trained/.
Those files were fit on every listing, so a score on this data would be
in-sample. The script retrains a fixed XGBoost on 80% of listings and
scores the other 20% with sklearn's r2_score.

Two targets are reported:

  median  — the area-median target the training script uses
            (cluster x bedrooms x month, medians fit on the training
            listings only)
  raw     — the listing's own seasonally scaled price or occupancy

Usage (from backend/):
    python -m scripts.eval_holdout
"""
import contextlib
import io
import sys
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from cities import CITIES
from models.feature_engineering import (
    load_and_clean_listings,
    prepare_monthly_data,
    smooth_targets,
    synthesize_calendar_monthly,
)


def _xgb():
    return XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        tree_method="hist",
        n_jobs=-1,
    )


def _metrics(y_true, pred):
    return float(r2_score(y_true, pred)), float(mean_absolute_error(y_true, pred))


def evaluate_city(city_key):
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_and_clean_listings(city_key)
        monthly = synthesize_calendar_monthly(df, city_key=city_key)
        X, y_price, y_occ, _cols, listing_ids, months = prepare_monthly_data(df, monthly)

    rng = np.random.default_rng(42)
    unique_ids = np.unique(listing_ids)
    rng.shuffle(unique_ids)
    n_test = max(1, int(round(0.2 * len(unique_ids))))
    test_ids = set(unique_ids[:n_test])
    test_mask = np.isin(listing_ids, list(test_ids))
    train_mask = ~test_mask

    with contextlib.redirect_stdout(io.StringIO()):
        y_price_s, y_occ_s, _n_smoothed, n_groups = smooth_targets(
            X, y_price, y_occ, listing_ids, months=months, train_mask=train_mask,
        )

    def predict(y_train):
        model = _xgb()
        model.fit(X.loc[train_mask], y_train[train_mask])
        return model.predict(X.loc[test_mask])

    pred_price_median = predict(y_price_s)
    pred_occ_median = predict(y_occ_s)
    pred_price_raw = predict(y_price)
    pred_occ_raw = predict(y_occ)

    yp, yo = y_price[test_mask], y_occ[test_mask]
    yps, yos = y_price_s[test_mask], y_occ_s[test_mask]

    price_med_r2, _ = _metrics(yps, pred_price_median)
    price_list_r2, price_list_mae = _metrics(yp, pred_price_median)
    occ_med_r2, _ = _metrics(yos, pred_occ_median)
    occ_list_r2, occ_list_mae = _metrics(yo, pred_occ_median)
    raw_price_r2, raw_price_mae = _metrics(yp, pred_price_raw)
    raw_occ_r2, raw_occ_mae = _metrics(yo, pred_occ_raw)

    return {
        "city": city_key,
        "listings": int(len(unique_ids)),
        "test_listings": int(n_test),
        "groups": int(n_groups),
        "price_r2_median_target": price_med_r2,
        "price_r2_listing": price_list_r2,
        "price_mae_listing": price_list_mae,
        "occ_r2_median_target": occ_med_r2,
        "occ_r2_listing": occ_list_r2,
        "occ_mae_listing": occ_list_mae,
        "raw_model_price_r2": raw_price_r2,
        "raw_model_price_mae": raw_price_mae,
        "raw_model_occ_r2": raw_occ_r2,
        "raw_model_occ_mae": raw_occ_mae,
    }


def main():
    cities = [a for a in sys.argv[1:] if not a.startswith("-")] or list(CITIES)
    header = (
        f"{'city':16} {'n':>6} "
        f"{'P_med':>7} {'P_list':>7} {'P_mae':>7} "
        f"{'O_med':>7} {'O_list':>7} {'O_mae':>7} "
        f"{'P_raw':>7} {'O_raw':>7}"
    )
    print(header)
    t0 = time.time()
    for city in cities:
        if city not in CITIES:
            print(f"Unknown city: {city}", file=sys.stderr)
            return 1
        row = evaluate_city(city)
        print(
            f"{row['city']:16} {row['listings']:6d} "
            f"{row['price_r2_median_target']:7.3f} {row['price_r2_listing']:7.3f} "
            f"{row['price_mae_listing']:7.1f} "
            f"{row['occ_r2_median_target']:7.3f} {row['occ_r2_listing']:7.3f} "
            f"{row['occ_mae_listing']:7.3f} "
            f"{row['raw_model_price_r2']:7.3f} {row['raw_model_occ_r2']:7.3f}"
        )
    print(f"\n{time.time() - t0:.1f}s  "
          f"P_med/O_med = R² vs area-median target (what training fits). "
          f"P_list/O_list = that same model vs the listing's own rate. "
          f"P_raw/O_raw = a model trained on the listing rate itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
