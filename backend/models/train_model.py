"""
Train ML models for STR investment prediction.

Uses LightGBM and XGBoost — the two best models for this tabular
regression task.  Both handle NaN natively (important for optional
physical features like sqft), are histogram-based (fast), and
consistently outperform linear/random-forest on this data.

Compares via RandomizedSearchCV (5-fold CV), selects the winner by
lowest mean CV MSE, and reports final metrics on a held-out test set.

Usage: python -m models.train_model [city_key] [--fast|--quick]
"""
import os
import sys
import time
import warnings
import joblib
import numpy as np
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Add backend dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.feature_engineering import (
    load_and_clean_listings,
    prepare_features,
    load_calendar_monthly,
    prepare_monthly_data,
    smooth_targets,
)

# Suppress convergence / future warnings during search
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Optional imports — script works without them
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ---------------------------------------------------------------------------
# Model definitions + hyperparameter distributions
# ---------------------------------------------------------------------------

def _build_candidates():
    """Return list of (name, estimator, param_distributions) tuples.

    Only LightGBM and XGBoost — the two best models for tabular regression
    with optional NaN features.  Falls back to sklearn GradientBoosting
    only if neither library is installed.
    """
    candidates = []

    # 1. LightGBM (preferred — fastest, best NaN handling)
    if HAS_LGB:
        candidates.append((
            "LightGBM",
            LGBMRegressor(random_state=42, verbosity=-1),
            {
                "n_estimators": sp_stats.randint(200, 801),
                "max_depth": sp_stats.randint(4, 12),
                "learning_rate": sp_stats.uniform(0.01, 0.19),
                "subsample": sp_stats.uniform(0.7, 0.3),
                "colsample_bytree": sp_stats.uniform(0.5, 0.5),
                "num_leaves": sp_stats.randint(20, 128),
                "min_child_samples": sp_stats.randint(5, 51),
                "reg_alpha": sp_stats.uniform(0, 1),
                "reg_lambda": sp_stats.uniform(0, 1),
            },
        ))

    # 2. XGBoost (strong second — histogram-based, native NaN)
    if HAS_XGB:
        candidates.append((
            "XGBoost",
            XGBRegressor(
                random_state=42,
                verbosity=0,
                tree_method="hist",
            ),
            {
                "n_estimators": sp_stats.randint(200, 801),
                "max_depth": sp_stats.randint(4, 12),
                "learning_rate": sp_stats.uniform(0.01, 0.19),
                "subsample": sp_stats.uniform(0.7, 0.3),
                "colsample_bytree": sp_stats.uniform(0.5, 0.5),
                "reg_alpha": sp_stats.uniform(0, 1),
                "reg_lambda": sp_stats.uniform(0.5, 1.5),
                "min_child_weight": sp_stats.randint(1, 11),
            },
        ))

    # 3. Fallback: sklearn GradientBoosting (no native NaN, slower)
    if not candidates:
        candidates.append((
            "GradientBoosting",
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": sp_stats.randint(200, 601),
                "max_depth": sp_stats.randint(4, 10),
                "learning_rate": sp_stats.uniform(0.01, 0.19),
                "subsample": sp_stats.uniform(0.7, 0.3),
                "min_samples_split": sp_stats.randint(2, 21),
                "min_samples_leaf": sp_stats.randint(1, 11),
            },
        ))

    return candidates


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------
N_ITER = 50
CV_FOLDS = 5
FAST_MODE = False
QUICK_MODE = False


def _run_search(name, estimator, param_dist, X_train, y_train):
    """Run RandomizedSearchCV for a single model. Returns the fitted search."""
    n_iter = 8 if FAST_MODE else N_ITER
    cv = 3 if FAST_MODE else CV_FOLDS
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=min(n_iter, _max_combinations(param_dist)),
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def _max_combinations(param_dist):
    """Rough upper-bound on grid size (for discrete-only grids like Ridge)."""
    n = 1
    for v in param_dist.values():
        if isinstance(v, list):
            n *= len(v)
        else:
            return N_ITER  # continuous distribution → use full n_iter
    return n


def _evaluate(model, X_test, y_test):
    """Return dict of test-set metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }


def _print_table(results, target_name):
    """Pretty-print comparison table."""
    header = f"{'Model':<20} {'CV MSE':>12} {'Test MSE':>12} {'Test RMSE':>12} {'Test MAE':>12} {'Test R²':>10}"
    print(f"\n{'=' * len(header)}")
    print(f"  Results for: {target_name}")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<20} "
            f"{r['cv_mse']:>12.4f} "
            f"{r['test']['MSE']:>12.4f} "
            f"{r['test']['RMSE']:>12.4f} "
            f"{r['test']['MAE']:>12.4f} "
            f"{r['test']['R2']:>10.4f}"
        )
    print("-" * len(header))


def _quick_train(target_name, X_train, X_test, y_train, y_test):
    """Train a single model with good defaults — no search, no CV."""
    if HAS_LGB:
        print(f"  [{target_name}] Quick-training LightGBM...")
        t0 = time.time()
        model = LGBMRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            num_leaves=63, min_child_samples=20,
            random_state=42, verbosity=-1, n_jobs=-1,
        )
    elif HAS_XGB:
        print(f"  [{target_name}] Quick-training XGBoost...")
        t0 = time.time()
        model = XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, tree_method="hist", n_jobs=-1,
        )
    else:
        print(f"  [{target_name}] Quick-training GradientBoosting...")
        t0 = time.time()
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, min_samples_split=5, random_state=42,
        )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    metrics = _evaluate(model, X_test, y_test)
    print(f"    done in {elapsed:.1f}s  |  Test R²={metrics['R2']:.4f}")
    return model


def train_target(target_name, X_train, X_test, y_train, y_test):
    """Compare candidate models for one target. Return best model."""
    if QUICK_MODE:
        return _quick_train(target_name, X_train, X_test, y_train, y_test)

    candidates = _build_candidates()
    if FAST_MODE:
        # In fast mode, just use the first (best) candidate
        candidates = candidates[:1]
    results = []

    for name, estimator, param_dist in candidates:
        n_iter_display = 8 if FAST_MODE else N_ITER
        cv_display = 3 if FAST_MODE else CV_FOLDS
        print(f"  [{target_name}] Searching {name} ({n_iter_display} iters, {cv_display}-fold CV)...")
        t0 = time.time()
        search = _run_search(name, estimator, param_dist, X_train, y_train)
        elapsed = time.time() - t0

        cv_mse = -search.best_score_  # neg_mean_squared_error → positive
        test_metrics = _evaluate(search.best_estimator_, X_test, y_test)

        results.append({
            "name": name,
            "cv_mse": cv_mse,
            "test": test_metrics,
            "model": search.best_estimator_,
            "best_params": search.best_params_,
        })
        print(f"    done in {elapsed:.1f}s  |  CV MSE={cv_mse:.4f}  |  Test R²={test_metrics['R2']:.4f}")

    # Select winner by lowest CV MSE (not test MSE)
    results.sort(key=lambda r: r["cv_mse"])
    _print_table(results, target_name)

    winner = results[0]
    print(f"\n  >>> Best model for {target_name}: {winner['name']}  (CV MSE={winner['cv_mse']:.4f})")
    print(f"      Best params: {winner['best_params']}")
    return winner["model"]


def train_models(city_key=None):
    city_key = city_key or 'austin'
    print(f"Loading and cleaning Inside Airbnb data for {city_key}...")
    df = load_and_clean_listings(city_key=city_key)
    print(f"  {len(df)} listings after cleaning")

    print("Preparing features...")
    X, targets, feature_cols, listing_ids = prepare_features(df)
    print(f"  {len(feature_cols)} features: {feature_cols[:5]}...")

    avail = []
    if HAS_XGB:
        avail.append("XGBoost")
    if HAS_LGB:
        avail.append("LightGBM")
    if avail:
        print(f"  ML libraries: {', '.join(avail)}")
    else:
        print("  WARNING: Install lightgbm and/or xgboost for best results")
        print("           Falling back to sklearn GradientBoosting (slower, no native NaN)")

    models_dir = config.city_models_dir(city_key)

    # Save feature metadata (needed by prediction_service)
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_columns.joblib"))
    medians = X.median().to_dict()
    joblib.dump(medians, os.path.join(models_dir, "feature_medians.joblib"))

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
    }, os.path.join(models_dir, "feature_stats.joblib"))
    print(f"  Saved feature_stats.joblib ({len(feature_stats)} features, {len(X)} samples, coords={'yes' if training_coords else 'no'})")

    overall_start = time.time()

    # ---- Load calendar data for monthly models ----
    print("\nLoading calendar data for monthly models...")
    calendar_monthly = load_calendar_monthly(city_key=city_key)
    use_monthly = False
    X_monthly = None

    MIN_MONTHLY_ROWS = 500  # Need enough data for monthly models to generalize
    monthly_listing_ids = None
    monthly_months = None

    if calendar_monthly is None:
        print("  No calendar data found — generating synthetic monthly data...")
        from models.feature_engineering import synthesize_calendar_monthly
        calendar_monthly = synthesize_calendar_monthly(df, city_key=city_key)

    if calendar_monthly is not None:
        print(f"  Calendar data loaded: {len(calendar_monthly)} listing-month rows")
        X_monthly, y_price_monthly, y_occ_monthly, monthly_feature_cols, monthly_listing_ids, monthly_months = prepare_monthly_data(df, calendar_monthly)
        if X_monthly is not None and len(X_monthly) >= MIN_MONTHLY_ROWS:
            use_monthly = True
            n_listings = len(np.unique(monthly_listing_ids))
            print(f"  Monthly dataset: {len(X_monthly)} rows from {n_listings} listings, {len(monthly_feature_cols)} features")
            print(f"  (includes sin_month, cos_month for seasonality)")
            print(f"  Split: by listing ID (all months of a listing stay in train OR test)")
        elif X_monthly is not None:
            print(f"  Monthly dataset too small ({len(X_monthly)} rows < {MIN_MONTHLY_ROWS}), using listing-level data")

    # Clean up stale monthly files if not using monthly models
    if not use_monthly:
        for stale in ['monthly_feature_columns.joblib', 'monthly_feature_medians.joblib']:
            stale_path = os.path.join(models_dir, stale)
            if os.path.exists(stale_path):
                os.remove(stale_path)
                print(f"  Removed stale {stale}")

    # ---- Split FIRST, then smooth (prevents target leakage) ----
    if use_monthly:
        # Split by listing ID — all months of a listing go to train OR test
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx_train, idx_test = next(gss.split(X_monthly, groups=monthly_listing_ids))
        n_train_listings = len(np.unique(monthly_listing_ids[idx_train]))
        n_test_listings = len(np.unique(monthly_listing_ids[idx_test]))
        print(f"\n  Split: {len(idx_train)} train rows ({n_train_listings} listings) | {len(idx_test)} test rows ({n_test_listings} listings)")

        # Smooth using ONLY training data — test targets come from train-derived medians
        train_mask = np.zeros(len(X_monthly), dtype=bool)
        train_mask[idx_train] = True
        y_price_sm, y_occ_sm, _, _ = smooth_targets(
            X_monthly, y_price_monthly, y_occ_monthly, monthly_listing_ids,
            months=monthly_months, train_mask=train_mask)

        # Save monthly metadata
        joblib.dump(monthly_feature_cols, os.path.join(models_dir, "monthly_feature_columns.joblib"))
        monthly_medians = X_monthly.median().to_dict()
        joblib.dump(monthly_medians, os.path.join(models_dir, "monthly_feature_medians.joblib"))

    elif "price" in targets:
        idx_train, idx_test = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42
        )
        print(f"\n  Split: {len(idx_train)} train | {len(idx_test)} test (listing-level)")

        # Smooth using ONLY training data
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[idx_train] = True
        y_price_sm, y_occ_sm, _, _ = smooth_targets(
            X, targets['price'], targets.get('occupancy', np.zeros(len(X))),
            listing_ids, train_mask=train_mask)

    # ---- Nightly price ----
    print("\n" + "=" * 60)
    print("TARGET: nightly price")
    print("=" * 60)

    if use_monthly:
        print("  Training on monthly price data (seasonal, train-only smoothing)")
        X_train, X_test = X_monthly.iloc[idx_train], X_monthly.iloc[idx_test]
        y_train, y_test = y_price_sm[idx_train], y_price_sm[idx_test]

        best_price = train_target("price", X_train, X_test, y_train, y_test)
        joblib.dump(best_price, os.path.join(models_dir, "price_model.joblib"))
        print("  Saved price_model.joblib (month-aware)")

    elif "price" in targets:
        print("  No calendar data -- training on listing-level price (train-only smoothing)")
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y_price_sm[idx_train], y_price_sm[idx_test]

        best_price = train_target("price", X_train, X_test, y_train, y_test)
        joblib.dump(best_price, os.path.join(models_dir, "price_model.joblib"))
        print("  Saved price_model.joblib (no seasonality)")

    # ---- Occupancy rate ----
    print("\n" + "=" * 60)
    print("TARGET: occupancy rate")
    print("=" * 60)

    if use_monthly:
        print("  Training on monthly occupancy data (seasonal, train-only smoothing)")
        X_train, X_test = X_monthly.iloc[idx_train], X_monthly.iloc[idx_test]
        y_train, y_test = y_occ_sm[idx_train], y_occ_sm[idx_test]

        best_occ = train_target("occupancy", X_train, X_test, y_train, y_test)
        joblib.dump(best_occ, os.path.join(models_dir, "occupancy_model.joblib"))
        print("  Saved occupancy_model.joblib (month-aware)")

    elif "occupancy" in targets:
        print("  No calendar data -- training on listing-level occupancy (train-only smoothing)")
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y_occ_sm[idx_train], y_occ_sm[idx_test]

        best_occ = train_target("occupancy", X_train, X_test, y_train, y_test)
        joblib.dump(best_occ, os.path.join(models_dir, "occupancy_model.joblib"))
        print("  Saved occupancy_model.joblib (no seasonality)")

    total = time.time() - overall_start
    print(f"\nTraining complete! Total time: {total / 60:.1f} minutes")


if __name__ == "__main__":
    args = sys.argv[1:]
    if '--quick' in args:
        QUICK_MODE = True
        print("*** QUICK MODE: single GradientBoosting, no search ***\n")
        args = [a for a in args if a != '--quick']
    elif '--fast' in args:
        FAST_MODE = True
        print("*** FAST MODE: 8 iters, 3-fold CV, top 2 models only ***\n")
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
