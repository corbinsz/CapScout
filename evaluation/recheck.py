"""Training-cell recheck of the shipped nightly-rate specification.

The published model was chosen on an earlier column list. This command asks
two questions on the training cells only, with the same 2% rules as the study:

1. Do three extra columns, written down here before the scores are read,
   cut the equal-city ridge error by at least 2%?
2. On the columns that remain, which eligible model is the simplest within
   2% of the best grouped cross-validation error?

It does not score the frozen holdout. A candidate that fails the training-cell
bar is not a new published number.

    python -m evaluation.recheck

The three extra columns are the log of distance to downtown, guests times a
private-room flag, and bathrooms per guest. None of them uses the nightly rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from data.cities import STUDY_CITIES
from features.places import added_columns
from sklearn.model_selection import GroupKFold

from evaluation.study_models import cv_mae, make_forest, make_linear, make_ridge, make_xgb
from evaluation.study_protocol import (
    AMENITY_COLUMNS,
    FEATURE_MARGIN,
    HOME_COLUMNS,
    N_FOLDS,
    REVIEW_COLUMNS,
    choose_params,
    choose_simplest,
    improves_by,
)

# Alphas already chosen on the training cells, before the place-feature bundle.
# They are not searched again for the reproduction or the extra-column check.
FROZEN_RIDGE_ALPHA = {"austin": 100.0, "san-diego": 100.0, "seattle": 10.0}

# The published equal-city training-cell error of the shipped ridge model.
PUBLISHED_CV_MAE = 56.01474200828854

# Stop if the reproduction is not the published fit. A drifted frame should
# not be allowed to adopt a new model.
REPRODUCTION_TOLERANCE = 0.5

EXTRA_COLUMNS = (
    "log_distance_to_center",
    "accommodates_x_private",
    "baths_per_guest",
)


def shipped_columns(city_key: str) -> list[str]:
    """Columns in the app artifact for one city. Coast is omitted inland."""
    return [*HOME_COLUMNS, *AMENITY_COLUMNS, *REVIEW_COLUMNS, *added_columns(city_key)]


def with_extra_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the three pre-registered columns. Does not read the nightly rate."""
    out = frame.copy()
    distance = pd.to_numeric(out["distance_to_center_km"], errors="coerce")
    out["log_distance_to_center"] = np.log1p(distance.to_numpy(dtype=float))
    baths = pd.to_numeric(out["bathrooms"], errors="coerce").to_numpy(dtype=float)
    guests = pd.to_numeric(out["accommodates"], errors="coerce").to_numpy(dtype=float)
    per_guest = np.full(len(out), np.nan)
    known = np.isfinite(baths) & np.isfinite(guests) & (guests > 0)
    per_guest[known] = baths[known] / guests[known]
    out["baths_per_guest"] = per_guest
    return out


def training_folds(
    frame: pd.DataFrame, train_idx: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The same grouped folds the study uses on the training cells."""
    train_idx = np.asarray(train_idx, dtype=int)
    groups = frame["cell"].to_numpy()[train_idx]
    n_splits = min(N_FOLDS, int(len(np.unique(groups))))
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_pos, valid_pos in splitter.split(np.zeros(len(train_idx)), groups=groups):
        folds.append((train_idx[train_pos], train_idx[valid_pos]))
    return folds


def _equal_city(by_city: dict[str, float]) -> float:
    return float(np.mean([by_city[city] for city in STUDY_CITIES]))


def reproduce_ridge(frames: dict[str, pd.DataFrame], splits: dict) -> dict[str, float]:
    """Shipped columns, frozen alphas. This is the number the bundle was scored against."""
    scores = {}
    for city in STUDY_CITIES:
        columns = shipped_columns(city)
        alpha = FROZEN_RIDGE_ALPHA[city]
        y = frames[city]["price_clean"].to_numpy(dtype=float)

        def _ridge(columns=columns, alpha=alpha):
            return make_ridge(columns, alpha, kind="log", high=None)

        scores[city] = cv_mae(_ridge, frames[city], y, splits[city]["folds"])
    return scores


def extra_ridge(frames: dict[str, pd.DataFrame], splits: dict) -> dict[str, float]:
    """Same ridge, plus the three extra columns."""
    scores = {}
    for city in STUDY_CITIES:
        columns = [*shipped_columns(city), *EXTRA_COLUMNS]
        alpha = FROZEN_RIDGE_ALPHA[city]
        y = frames[city]["price_clean"].to_numpy(dtype=float)

        def _ridge(columns=columns, alpha=alpha):
            return make_ridge(columns, alpha, kind="log", high=None)

        scores[city] = cv_mae(_ridge, frames[city], y, splits[city]["folds"])
    return scores


def compare_models(
    frames: dict[str, pd.DataFrame], splits: dict, columns_for: dict[str, list[str]]
) -> dict[str, dict[str, float]]:
    """Small pre-registered grids. Cross-validation only. No holdout."""
    tuned: dict[str, dict[str, float]] = {city: {} for city in STUDY_CITIES}
    for city in STUDY_CITIES:
        frame = frames[city]
        columns = columns_for[city]
        y = frame["price_clean"].to_numpy(dtype=float)
        folds = splits[city]["folds"]
        print(f"  {city}: linear and ridge", flush=True)
        tuned[city]["linear"] = cv_mae(
            lambda columns=columns: make_linear(columns, kind="log", high=None),
            frame,
            y,
            folds,
        )
        ridge_trials = []
        for alpha in (100.0, 10.0, 1.0, 0.1):

            def _ridge(alpha=alpha, columns=columns):
                return make_ridge(columns, alpha, kind="log", high=None)

            mae = cv_mae(_ridge, frame, y, folds)
            ridge_trials.append({"mae": mae, "simplicity": [-alpha], "params": {"alpha": alpha}})
        tuned[city]["ridge"] = float(choose_params(ridge_trials)["mae"])
        print(f"    ridge {tuned[city]['ridge']:.2f}", flush=True)

        forest_trials = []
        for depth in (8, 16):
            for leaf in (20, 5):

                def _forest(depth=depth, leaf=leaf, columns=columns):
                    return make_forest(columns, depth, leaf, kind="log", high=None)

                mae = cv_mae(_forest, frame, y, folds)
                forest_trials.append(
                    {
                        "mae": mae,
                        "simplicity": [depth, -leaf],
                        "params": {"max_depth": depth, "min_samples_leaf": leaf},
                    }
                )
                print(f"    forest depth {depth} leaf {leaf} {mae:.2f}", flush=True)
        tuned[city]["random_forest"] = float(choose_params(forest_trials)["mae"])

        xgb_trials = []
        for depth in (4, 6):
            for child in (20.0, 5.0):

                def _xgb(depth=depth, child=child, columns=columns):
                    return make_xgb(columns, depth, child, kind="log", high=None)

                mae = cv_mae(_xgb, frame, y, folds)
                xgb_trials.append(
                    {
                        "mae": mae,
                        "simplicity": [depth, -child],
                        "params": {"max_depth": depth, "min_child_weight": child},
                    }
                )
                print(f"    xgb depth {depth} child {child} {mae:.2f}", flush=True)
        tuned[city]["xgboost"] = float(choose_params(xgb_trials)["mae"])
    return tuned


def load_training() -> tuple[dict[str, pd.DataFrame], dict]:
    """Cleaned frames and the frozen training folds. The test rows are not scored."""
    from evaluation.study_data import prepare_city
    from evaluation.study_split import PRIMARY_SEED, split_frame

    frames: dict[str, pd.DataFrame] = {}
    splits: dict = {}
    for city in STUDY_CITIES:
        frame, _report = prepare_city(city)
        frames[city] = with_extra_columns(frame)
        split = split_frame(frames[city], city, PRIMARY_SEED)
        splits[city] = {
            "train_idx": split["train_idx"],
            "folds": training_folds(frames[city], split["train_idx"]),
        }
    return frames, splits


def main() -> None:
    print("Loading the frozen training cells. The holdout is not scored.", flush=True)
    frames, splits = load_training()
    current = reproduce_ridge(frames, splits)
    current_mean = _equal_city(current)
    print(
        "Shipped ridge CV "
        + ", ".join(f"{city} ${current[city]:.2f}" for city in STUDY_CITIES)
        + f", equal-city ${current_mean:.2f} (published ${PUBLISHED_CV_MAE:.2f})",
        flush=True,
    )
    if abs(current_mean - PUBLISHED_CV_MAE) > REPRODUCTION_TOLERANCE:
        print(
            "Reproduction is outside "
            f"${REPRODUCTION_TOLERANCE:.2f} of the published training-cell error. "
            "No candidate is adopted.",
            flush=True,
        )
        return

    extra = extra_ridge(frames, splits)
    extra_mean = _equal_city(extra)
    cleared = improves_by(current_mean, extra_mean, FEATURE_MARGIN)
    print(
        "Extra columns CV "
        + ", ".join(f"{city} ${extra[city]:.2f}" for city in STUDY_CITIES)
        + f", equal-city ${extra_mean:.2f}. "
        + ("Clears 2%." if cleared else "Does not clear 2%."),
        flush=True,
    )
    columns_for = {
        city: [*shipped_columns(city), *(EXTRA_COLUMNS if cleared else ())] for city in STUDY_CITIES
    }
    print("Eligible models on the training cells", flush=True)
    tuned = compare_models(frames, splits, columns_for)
    means = {
        name: _equal_city({city: tuned[city][name] for city in STUDY_CITIES})
        for name in ("linear", "ridge", "random_forest", "xgboost")
    }
    # Area median is not refit here. The published comparison already includes it,
    # and it is far enough from these models that it cannot win the 2% band.
    chosen = choose_simplest([(name, means[name]) for name in means])
    print(
        "Equal-city CV "
        + ", ".join(f"{name} ${means[name]:.2f}" for name in means)
        + f". Simplest within 2%: {chosen}.",
        flush=True,
    )
    if chosen == "ridge" and not cleared:
        print(
            "The shipped ridge specification stays. The holdout was not re-scored.",
            flush=True,
        )
    else:
        print(
            "A candidate cleared a training-cell bar. "
            "The holdout was not scored by this command.",
            flush=True,
        )


if __name__ == "__main__":
    main()
