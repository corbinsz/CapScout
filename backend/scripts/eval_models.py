"""
Evaluate trained models on held-out test sets for all cities.
Shows R², MAE, RMSE, and example predictions vs actuals.

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
)
from sklearn.model_selection import train_test_split
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
    print(f"  {city['display']} — Model Evaluation (20% held-out test set)")
    print(f"{'='*65}")

    # Load data
    df = load_and_clean_listings(city_key=city_key)
    X, targets, feature_cols = prepare_features(df)
    print(f"  Dataset: {len(df)} listings, {len(feature_cols)} features")

    # Check for monthly models
    monthly_cols_path = os.path.join(models_dir, 'monthly_feature_columns.joblib')
    use_monthly = os.path.exists(monthly_cols_path)

    if use_monthly:
        calendar = load_calendar_monthly(city_key=city_key)
        X_m, y_price_m, y_occ_m, m_cols = prepare_monthly_data(df, calendar)
        if X_m is not None and len(X_m) > 0:
            print(f"  Monthly data: {len(X_m)} listing-month rows")
        else:
            use_monthly = False

    # --- Price model ---
    price_model = joblib.load(price_path)
    model_name = type(price_model).__name__
    print(f"\n  Price Model: {model_name}")

    if use_monthly:
        idx_train, idx_test = train_test_split(
            np.arange(len(X_m)), test_size=0.2, random_state=42
        )
        X_test = X_m.iloc[idx_test]
        y_test = y_price_m[idx_test]
    else:
        idx_train, idx_test = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42
        )
        X_test = X.iloc[idx_test]
        y_test = targets['price'][idx_test]

    y_pred = price_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1, None))) * 100

    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*37}")
    print(f"  {'R² (variance explained)':<25} {r2:>10.4f}")
    print(f"  {'RMSE':<25} {'$'+f'{rmse:.2f}':>10}")
    print(f"  {'MAE (avg error)':<25} {'$'+f'{mae:.2f}':>10}")
    print(f"  {'MAPE (avg % error)':<25} {mape:>9.1f}%")
    print(f"  {'Test samples':<25} {len(y_test):>10}")

    # Show some examples
    print(f"\n  Sample predictions (nightly rate):")
    print(f"  {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    indices = np.random.RandomState(42).choice(len(y_test), min(8, len(y_test)), replace=False)
    for i in indices:
        actual = y_test[i] if isinstance(y_test, np.ndarray) else y_test.iloc[i]
        pred = y_pred[i]
        err = pred - actual
        print(f"  ${actual:>8.2f}  ${pred:>8.2f}  {'+' if err >= 0 else ''}{err:>8.2f}")

    # --- Occupancy model ---
    if os.path.exists(occ_path):
        occ_model = joblib.load(occ_path)
        occ_name = type(occ_model).__name__
        print(f"\n  Occupancy Model: {occ_name}")

        if use_monthly:
            y_test_occ = y_occ_m[idx_test]
        else:
            y_test_occ = targets['occupancy'][idx_test]

        y_pred_occ = occ_model.predict(X_test)
        # Clip to [0, 1]
        y_pred_occ = np.clip(y_pred_occ, 0, 1)

        r2_occ = r2_score(y_test_occ, y_pred_occ)
        rmse_occ = np.sqrt(mean_squared_error(y_test_occ, y_pred_occ))
        mae_occ = mean_absolute_error(y_test_occ, y_pred_occ)

        print(f"  {'Metric':<25} {'Value':>10}")
        print(f"  {'-'*37}")
        print(f"  {'R² (variance explained)':<25} {r2_occ:>10.4f}")
        print(f"  {'RMSE':<25} {rmse_occ:>10.4f}")
        print(f"  {'MAE (avg error)':<25} {mae_occ:>10.4f}")
        print(f"  {'Test samples':<25} {len(y_test_occ):>10}")

        print(f"\n  Sample predictions (occupancy rate):")
        print(f"  {'Actual':>10} {'Predicted':>10} {'Error':>10}")
        for i in indices[:8]:
            actual = y_test_occ[i] if isinstance(y_test_occ, np.ndarray) else y_test_occ.iloc[i]
            pred = y_pred_occ[i]
            err = pred - actual
            print(f"  {actual:>9.1%}  {pred:>9.1%}  {'+' if err >= 0 else ''}{err:>9.1%}")

    # --- Summary interpretation ---
    print(f"\n  Interpretation:")
    if r2 >= 0.7:
        print(f"    Price model: GOOD — explains {r2*100:.0f}% of price variance, avg error ${mae:.0f}/night")
    elif r2 >= 0.4:
        print(f"    Price model: FAIR — explains {r2*100:.0f}% of variance, use with caution")
    else:
        print(f"    Price model: WEAK — only explains {r2*100:.0f}% of variance, predictions unreliable")

    if os.path.exists(occ_path):
        if r2_occ >= 0.5:
            print(f"    Occupancy model: GOOD — explains {r2_occ*100:.0f}% of occupancy variance")
        elif r2_occ >= 0.2:
            print(f"    Occupancy model: FAIR — explains {r2_occ*100:.0f}%, occupancy is harder to predict")
        else:
            print(f"    Occupancy model: WEAK — only {r2_occ*100:.0f}%, occupancy highly variable")


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    cities = args if args else list(CITIES.keys())

    for ck in cities:
        if ck in CITIES:
            eval_city(ck)

    print(f"\n{'='*65}")
    print("  Done!")
    print(f"{'='*65}")
