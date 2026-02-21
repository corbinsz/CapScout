"""
Train XGBoost models for STR investment prediction.

XGBoost only -- appropriate for our data sizes (40K-156K rows),
handles NaN natively, and histogram-based for fast training.

Trains on 100% of Airbnb data (no hold-out split). CV folds handle
evaluation; Redfin for-sale properties are the real test set.

Reports both Train R2 (fit) and CV R2 (generalization) after training.

Usage: python -m models.train_model [city_key] [--fast|--quick|--all]
"""
import os
import sys
import time
import warnings
import joblib
import numpy as np
from scipy import stats as sp_stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Add backend dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.feature_engineering import (
    load_and_clean_listings,
    prepare_features,
    prepare_monthly_data,
    smooth_targets,
)

# Suppress convergence / future warnings during search
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# XGBoost hyperparameter search space
# ---------------------------------------------------------------------------
PARAM_DIST = {
    "n_estimators": sp_stats.randint(200, 801),
    "max_depth": sp_stats.randint(4, 13),
    "learning_rate": sp_stats.uniform(0.01, 0.19),
    "subsample": sp_stats.uniform(0.7, 0.3),
    "colsample_bytree": sp_stats.uniform(0.5, 0.5),
    "reg_alpha": sp_stats.uniform(0, 1),
    "reg_lambda": sp_stats.uniform(0.5, 1.5),
    "min_child_weight": sp_stats.randint(1, 11),
}

# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------
N_ITER = 50
CV_FOLDS = 5
FAST_MODE = False
QUICK_MODE = False


def _quick_train(target_name, X_all, y_all):
    """Train XGBoost with good defaults -- no search, no CV."""
    print(f"  [{target_name}] Quick-training XGBoost on {len(X_all)} rows...")
    t0 = time.time()
    model = XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, tree_method="hist", n_jobs=-1,
    )
    model.fit(X_all, y_all)
    elapsed = time.time() - t0

    train_r2 = r2_score(y_all, model.predict(X_all))
    print(f"    done in {elapsed:.1f}s  |  Train R2={train_r2:.4f}  (no CV in quick mode)")
    return model


def train_target(target_name, X_all, y_all):
    """Train XGBoost with RandomizedSearchCV on all data.

    refit=True means the final model is retrained on 100% of the data
    with the best hyperparameters found during CV.

    Reports:
      - CV R2: average across 5 held-out folds (generalization)
      - Train R2: predict on all training data with final model (fit)
    """
    if QUICK_MODE:
        return _quick_train(target_name, X_all, y_all)

    n_iter = 8 if FAST_MODE else N_ITER
    cv = 3 if FAST_MODE else CV_FOLDS

    print(f"  [{target_name}] Searching XGBoost ({n_iter} iters, {cv}-fold CV, {len(X_all)} rows)...")
    t0 = time.time()

    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0, tree_method="hist"),
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
        error_score="raise",
    )
    search.fit(X_all, y_all)
    elapsed = time.time() - t0

    # CV R2 (held-out fold average)
    cv_mse = -search.best_score_
    cv_r2 = 1 - cv_mse / np.var(y_all)

    # Train R2 (final refitted model on all data)
    model = search.best_estimator_
    train_r2 = r2_score(y_all, model.predict(X_all))

    print(f"    done in {elapsed:.1f}s")
    print(f"    Train R2={train_r2:.4f}  |  CV R2={cv_r2:.4f}  |  gap={train_r2 - cv_r2:.4f}")
    print(f"    Best params: {search.best_params_}")

    return model


def train_models(city_key=None):
    city_key = city_key or 'austin'
    print(f"Loading and cleaning Inside Airbnb data for {city_key}...")
    df = load_and_clean_listings(city_key=city_key)
    print(f"  {len(df)} listings after cleaning")

    print("Preparing features...")
    X, targets, feature_cols, listing_ids = prepare_features(df)
    print(f"  {len(feature_cols)} features: {feature_cols}")

    models_dir = config.city_models_dir(city_key)

    # Save feature metadata (needed by prediction_service)
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_columns.joblib"), compress=3)
    medians = X.median().to_dict()
    joblib.dump(medians, os.path.join(models_dir, "feature_medians.joblib"), compress=3)

    # Save feature statistics for confidence scoring
    feature_stats = {}
    for col in feature_cols:
        if col in X.columns:
            s = X[col].dropna()
            feature_stats[col] = {
                'min': float(s.min()),
                'max': float(s.max()),
                'p5': float(s.quantile(0.05)),
                'p25': float(s.quantile(0.25)),
                'p50': float(s.quantile(0.50)),
                'p75': float(s.quantile(0.75)),
                'p95': float(s.quantile(0.95)),
                'mean': float(s.mean()),
                'std': float(s.std()),
            }
    target_stats = {}
    if 'price' in targets:
        target_stats['price_mean'] = float(np.mean(targets['price']))
        target_stats['price_std'] = float(np.std(targets['price']))
    if 'occupancy' in targets:
        target_stats['occ_mean'] = float(np.mean(targets['occupancy']))
        target_stats['occ_std'] = float(np.std(targets['occupancy']))
    target_stats['n_training'] = len(X)

    # Save training coordinates for geographic confidence scoring
    training_coords = None
    if 'latitude' in X.columns and 'longitude' in X.columns:
        training_coords = {
            'latitude': X['latitude'].values.astype(np.float32),
            'longitude': X['longitude'].values.astype(np.float32),
        }

    joblib.dump({
        'features': feature_stats,
        'targets': target_stats,
        'training_coords': training_coords,
    }, os.path.join(models_dir, "feature_stats.joblib"), compress=3)
    print(f"  Saved feature_stats.joblib ({len(feature_stats)} features, {len(X)} samples)")

    overall_start = time.time()

    # ---- Build monthly training data ----
    # Always use synthetic monthly data.  Inside Airbnb calendar occupancy is
    # contaminated by the booking-horizon artifact: "not available" can mean
    # booked OR host-blocked, and far-future months are almost all blocked,
    # creating fake 0% / 100% occupancy swings.  Synthetic data uses each
    # listing's availability_365 as base occupancy scaled by city-specific
    # seasonal factors -- reliable and consistent across all cities.
    print("\nBuilding synthetic monthly data...")
    from models.feature_engineering import synthesize_calendar_monthly
    calendar_monthly = synthesize_calendar_monthly(df, city_key=city_key)
    use_monthly = False
    X_monthly = None

    MIN_MONTHLY_ROWS = 500
    monthly_listing_ids = None
    monthly_months = None

    if calendar_monthly is not None:
        print(f"  Calendar data loaded: {len(calendar_monthly)} listing-month rows")
        X_monthly, y_price_monthly, y_occ_monthly, monthly_feature_cols, monthly_listing_ids, monthly_months = prepare_monthly_data(df, calendar_monthly)
        if X_monthly is not None and len(X_monthly) >= MIN_MONTHLY_ROWS:
            use_monthly = True
            n_listings = len(np.unique(monthly_listing_ids))
            print(f"  Monthly dataset: {len(X_monthly)} rows from {n_listings} listings, {len(monthly_feature_cols)} features")
        elif X_monthly is not None:
            print(f"  Monthly dataset too small ({len(X_monthly)} rows < {MIN_MONTHLY_ROWS}), using listing-level data")

    # Clean up stale monthly files if not using monthly models
    if not use_monthly:
        for stale in ['monthly_feature_columns.joblib', 'monthly_feature_medians.joblib']:
            stale_path = os.path.join(models_dir, stale)
            if os.path.exists(stale_path):
                os.remove(stale_path)
                print(f"  Removed stale {stale}")

    # ---- Smooth targets on ALL data ----
    if use_monthly:
        n_listings = len(np.unique(monthly_listing_ids))
        print(f"\n  Training on ALL {len(X_monthly)} rows ({n_listings} listings)")

        y_price_sm, y_occ_sm, _, _ = smooth_targets(
            X_monthly, y_price_monthly, y_occ_monthly, monthly_listing_ids,
            months=monthly_months)

        # Save monthly metadata
        joblib.dump(monthly_feature_cols, os.path.join(models_dir, "monthly_feature_columns.joblib"), compress=3)
        monthly_medians = X_monthly.median().to_dict()
        joblib.dump(monthly_medians, os.path.join(models_dir, "monthly_feature_medians.joblib"), compress=3)

    elif "price" in targets:
        print(f"\n  Training on ALL {len(X)} listings")

        y_price_sm, y_occ_sm, _, _ = smooth_targets(
            X, targets['price'], targets.get('occupancy', np.zeros(len(X))),
            listing_ids)

    # ---- Nightly price ----
    print("\n" + "=" * 60)
    print("TARGET: nightly price")
    print("=" * 60)

    if use_monthly:
        best_price = train_target("price", X_monthly, y_price_sm)
        joblib.dump(best_price, os.path.join(models_dir, "price_model.joblib"), compress=3)
        print("  Saved price_model.joblib (month-aware)")

    elif "price" in targets:
        best_price = train_target("price", X, y_price_sm)
        joblib.dump(best_price, os.path.join(models_dir, "price_model.joblib"), compress=3)
        print("  Saved price_model.joblib (no seasonality)")

    # ---- Occupancy rate ----
    print("\n" + "=" * 60)
    print("TARGET: occupancy rate")
    print("=" * 60)

    if use_monthly:
        best_occ = train_target("occupancy", X_monthly, y_occ_sm)
        joblib.dump(best_occ, os.path.join(models_dir, "occupancy_model.joblib"), compress=3)
        print("  Saved occupancy_model.joblib (month-aware)")

    elif "occupancy" in targets:
        best_occ = train_target("occupancy", X, y_occ_sm)
        joblib.dump(best_occ, os.path.join(models_dir, "occupancy_model.joblib"), compress=3)
        print("  Saved occupancy_model.joblib (no seasonality)")

    total = time.time() - overall_start
    print(f"\nTraining complete! Total time: {total / 60:.1f} minutes")


if __name__ == "__main__":
    args = sys.argv[1:]
    if '--quick' in args:
        QUICK_MODE = True
        print("*** QUICK MODE: single XGBoost, no search ***\n")
        args = [a for a in args if a != '--quick']
    elif '--fast' in args:
        FAST_MODE = True
        print("*** FAST MODE: 8 iters, 3-fold CV ***\n")
        args = [a for a in args if a != '--fast']
    if '--all' in args:
        from cities import CITIES
        for ck in CITIES:
            train_models(city_key=ck)
    elif args:
        for a in args:
            if not a.startswith('-'):
                train_models(city_key=a)
    else:
        train_models()
