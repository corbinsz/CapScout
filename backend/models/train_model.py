"""
Train ML models for STR investment prediction.

Usage: python -m models.train_model  (from backend/ directory)
"""
import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Add backend dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.feature_engineering import load_and_clean_listings, prepare_features


def train_models():
    print("Loading and cleaning Inside Airbnb data...")
    df = load_and_clean_listings()
    print(f"  {len(df)} listings after cleaning")

    print("Preparing features...")
    X, targets, feature_cols = prepare_features(df)
    print(f"  {len(feature_cols)} features: {feature_cols[:5]}...")

    # Save feature column names for inference
    joblib.dump(feature_cols, os.path.join(config.TRAINED_MODELS_DIR, 'feature_columns.joblib'))

    # Save median values for default feature imputation
    medians = X.median().to_dict()
    joblib.dump(medians, os.path.join(config.TRAINED_MODELS_DIR, 'feature_medians.joblib'))

    # Train nightly rate model
    if 'price' in targets:
        print("\nTraining nightly rate model...")
        y_price = targets['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y_price, test_size=0.2, random_state=42)

        price_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        price_model.fit(X_train, y_train)

        y_pred = price_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"  R-squared: {r2:.4f}")
        print(f"  MAE: ${mae:.2f}")

        joblib.dump(price_model, os.path.join(config.TRAINED_MODELS_DIR, 'price_model.joblib'))
        print("  Saved price_model.joblib")

    # Train occupancy rate model
    if 'occupancy' in targets:
        print("\nTraining occupancy rate model...")
        y_occ = targets['occupancy']
        X_train, X_test, y_train, y_test = train_test_split(X, y_occ, test_size=0.2, random_state=42)

        occ_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        occ_model.fit(X_train, y_train)

        y_pred = occ_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"  R-squared: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")

        joblib.dump(occ_model, os.path.join(config.TRAINED_MODELS_DIR, 'occupancy_model.joblib'))
        print("  Saved occupancy_model.joblib")

    print("\nTraining complete!")


if __name__ == '__main__':
    train_models()
