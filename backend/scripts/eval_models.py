"""
Evaluate trained models on held-out test sets for all cities.
Shows R², MAE, RMSE, and example predictions vs actuals.

Uses train-only target smoothing: clusters and medians are computed
from training data only, so test metrics reflect true out-of-sample
accuracy (no target leakage).

Usage: python -m scripts.eval_models          # all cities
       python -m scripts.eval_models austin   # one city
"""
import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from cities import CITIES
from models.feature_engineering import (
    load_and_clean_listings,
    prepare_features,
    load_calendar_monthly,
    prepare_monthly_data,
    smooth_targets,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eval_city(city_key):
    models_dir = config.city_models_dir(city_key)
    price_path = os.path.join(models_dir, 'price_model.joblib')
    occ_path = os.path.join(models_dir, 'occupancy_model.joblib')

    if not os.path.exists(price_path):
        print(f'\n{city_key}: no trained models found, skipping')
        return

    city = CITIES[city_key]
    print(f"\n{'='*65}")
    print(f"  {city['display']} -- Model Evaluation (20% held-out test set)")
    print(f"{'='*65}")

    # Load data
    df = load_and_clean_listings(city_key=city_key)
    X, targets, feature_cols, listing_ids = prepare_features(df)
    print(f"  Dataset: {len(df)} listings, {len(feature_cols)} features")

    # Check for monthly models
    monthly_cols_path = os.path.join(models_dir, 'monthly_feature_columns.joblib')
    use_monthly = os.path.exists(monthly_cols_path)

    monthly_listing_ids = None
    monthly_months = None
    if use_monthly:
        calendar = load_calendar_monthly(city_key=city_key)
        if calendar is None:
            from models.feature_engineering import synthesize_calendar_monthly
            calendar = synthesize_calendar_monthly(df, city_key=city_key)
        X_m, y_price_m, y_occ_m, m_cols, monthly_listing_ids, monthly_months = prepare_monthly_data(df, calendar)
        if X_m is not None and len(X_m) > 0:
            n_listings = len(np.unique(monthly_listing_ids))
            print(f"  Monthly data: {len(X_m)} listing-month rows from {n_listings} listings")
        else:
            use_monthly = False

    # --- Split FIRST, then smooth with train-only data ---
    if use_monthly:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx_train, idx_test = next(gss.split(X_m, groups=monthly_listing_ids))
        n_test_listings = len(np.unique(monthly_listing_ids[idx_test]))
        print(f"  Test set: {len(idx_test)} rows from {n_test_listings} unseen listings")

        train_mask = np.zeros(len(X_m), dtype=bool)
        train_mask[idx_train] = True
        y_price_sm, y_occ_sm, _, _ = smooth_targets(
            X_m, y_price_m, y_occ_m, monthly_listing_ids,
            months=monthly_months, train_mask=train_mask)

        X_test = X_m.iloc[idx_test]
        y_test_price = y_price_sm[idx_test]
        y_test_occ = y_occ_sm[idx_test]
    else:
        idx_train, idx_test = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42
        )
        print(f"  Test set: {len(idx_test)} listings")

        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[idx_train] = True
        y_price_sm, y_occ_sm, _, _ = smooth_targets(
            X, targets['price'], targets.get('occupancy', np.zeros(len(X))),
            listing_ids, train_mask=train_mask)

        X_test = X.iloc[idx_test]
        y_test_price = y_price_sm[idx_test]
        y_test_occ = y_occ_sm[idx_test]

    # --- Get train data for overfitting check ---
    if use_monthly:
        X_train = X_m.iloc[idx_train]
        y_train_price = y_price_sm[idx_train]
        y_train_occ = y_occ_sm[idx_train]
    else:
        X_train = X.iloc[idx_train]
        y_train_price = y_price_sm[idx_train]
        y_train_occ = y_occ_sm[idx_train]

    # --- Price model ---
    price_model = joblib.load(price_path)
    model_name = type(price_model).__name__
    print(f"\n  Price Model: {model_name}")

    y_pred = price_model.predict(X_test)
    r2 = r2_score(y_test_price, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_price, y_pred))
    mae = mean_absolute_error(y_test_price, y_pred)
    mape = np.mean(np.abs((y_test_price - y_pred) / np.clip(y_test_price, 1, None))) * 100

    # Train metrics for overfitting check
    y_pred_train = price_model.predict(X_train)
    r2_train = r2_score(y_train_price, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train_price, y_pred_train))

    print(f"  {'Metric':<25} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print(f"  {'-'*57}")
    print(f"  {'R2 (variance explained)':<25} {r2_train:>10.4f} {r2:>10.4f} {r2_train - r2:>+10.4f}")
    print(f"  {'RMSE':<25} {'$'+f'{rmse_train:.2f}':>10} {'$'+f'{rmse:.2f}':>10}")
    print(f"  {'MAE (avg error)':<25} {'':>10} {'$'+f'{mae:.2f}':>10}")
    print(f"  {'MAPE (avg % error)':<25} {'':>10} {mape:>9.1f}%")
    print(f"  {'Samples':<25} {len(y_train_price):>10} {len(y_test_price):>10}")

    # Show some examples
    print(f"\n  Sample predictions (nightly rate):")
    print(f"  {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    indices = np.random.RandomState(42).choice(len(y_test_price), min(8, len(y_test_price)), replace=False)
    for i in indices:
        actual = y_test_price[i]
        pred = y_pred[i]
        err = pred - actual
        print(f"  ${actual:>8.2f}  ${pred:>8.2f}  {'+' if err >= 0 else ''}{err:>8.2f}")

    # --- Occupancy model ---
    r2_occ = None
    if os.path.exists(occ_path):
        occ_model = joblib.load(occ_path)
        occ_name = type(occ_model).__name__
        print(f"\n  Occupancy Model: {occ_name}")

        y_pred_occ = np.clip(occ_model.predict(X_test), 0, 1)

        r2_occ = r2_score(y_test_occ, y_pred_occ)
        rmse_occ = np.sqrt(mean_squared_error(y_test_occ, y_pred_occ))
        mae_occ = mean_absolute_error(y_test_occ, y_pred_occ)

        # Train metrics for overfitting check
        y_pred_occ_train = np.clip(occ_model.predict(X_train), 0, 1)
        r2_occ_train = r2_score(y_train_occ, y_pred_occ_train)
        rmse_occ_train = np.sqrt(mean_squared_error(y_train_occ, y_pred_occ_train))

        print(f"  {'Metric':<25} {'Train':>10} {'Test':>10} {'Gap':>10}")
        print(f"  {'-'*57}")
        print(f"  {'R2 (variance explained)':<25} {r2_occ_train:>10.4f} {r2_occ:>10.4f} {r2_occ_train - r2_occ:>+10.4f}")
        print(f"  {'RMSE':<25} {rmse_occ_train:>10.4f} {rmse_occ:>10.4f}")
        print(f"  {'MAE (avg error)':<25} {'':>10} {mae_occ:>10.4f}")
        print(f"  {'Samples':<25} {len(y_train_occ):>10} {len(y_test_occ):>10}")

        print(f"\n  Sample predictions (occupancy rate):")
        print(f"  {'Actual':>10} {'Predicted':>10} {'Error':>10}")
        for i in indices[:8]:
            actual = y_test_occ[i]
            pred = y_pred_occ[i]
            err = pred - actual
            print(f"  {actual:>9.1%}  {pred:>9.1%}  {'+' if err >= 0 else ''}{err:>9.1%}")

    # --- Summary interpretation ---
    print(f"\n  Interpretation:")
    if r2 >= 0.7:
        print(f"    Price model: GOOD -- explains {r2*100:.0f}% of price variance, avg error ${mae:.0f}/night")
    elif r2 >= 0.4:
        print(f"    Price model: FAIR -- explains {r2*100:.0f}% of variance, use with caution")
    else:
        print(f"    Price model: WEAK -- only explains {r2*100:.0f}% of variance, predictions unreliable")

    if r2_occ is not None:
        if r2_occ >= 0.5:
            print(f"    Occupancy model: GOOD -- explains {r2_occ*100:.0f}% of occupancy variance")
        elif r2_occ >= 0.2:
            print(f"    Occupancy model: FAIR -- explains {r2_occ*100:.0f}%, occupancy is harder to predict")
        else:
            print(f"    Occupancy model: WEAK -- only {r2_occ*100:.0f}%, occupancy highly variable")


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    cities = args if args else list(CITIES.keys())

    for ck in cities:
        if ck in CITIES:
            eval_city(ck)

    print(f"\n{'='*65}")
    print("  Done!")
    print(f"{'='*65}")
