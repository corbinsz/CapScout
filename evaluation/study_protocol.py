"""Rules for this study, fixed before any model is fit.

The holdout is a spatial grid. Seeds 7, 11, 23, and 31 were already used
for published splits, and 0–4 and 42 were used in earlier development.
None of those seeds is reused here. The primary seed and the three
confirmation seeds were written down first. They were not searched.

What is allowed to choose a model
    Grouped cross-validation on the training cells chooses the feature
    prefix and the small hyperparameter grids. The primary holdout is scored
    once. The simplest eligible model within 2% of the best holdout MAE is
    the one that ships. The eligible models are the local median rate,
    linear regression, ridge regression, a random forest, and XGBoost.

What is not allowed
    The confirmation seeds are not used to pick a feature, a parameter, or
    a model. They are a second run of the same frozen choice. The nights
    bar is the same bar as the previous nights study: every city needs an
    R² of at least 0.10 and an average miss at least 10% lower than the
    local median nights. It is not lowered after the fact.
"""

from __future__ import annotations

from features.columns import CENSUS_COLUMNS

# Primes that are not 0–4, 7, 11, 23, 31, or 42. Not a search.
PRIMARY_SEED = 17
CONFIRM_SEEDS = (19, 29, 37)
FORBIDDEN_SEEDS = frozenset({0, 1, 2, 3, 4, 7, 11, 23, 31, 42})

CELL_KM = 3.0
HOLDOUT_FRACTION = 0.20
N_FOLDS = 5
BUFFER_KM = 1.0
MAX_KM_FROM_CENTER = 60.0

MIN_PRICE = 25.0
MAX_PRICE = 2000.0
MAX_BEDROOMS = 10
MAX_BATHROOMS = 12.0
MAX_GUESTS = 16
MAX_MINIMUM_NIGHTS = 90

# A neighbourhood x room-type median is used only when at least this many
# training listings support it. Not tuned.
BASELINE_MIN_COUNT = 8

# Feature prefixes and model families: simplest within this fraction of the
# best error wins. Hyperparameter ties use the tighter band.
FEATURE_MARGIN = 0.02
MODEL_MARGIN = 0.02
PARAM_MARGIN = 0.01

# Same nights bar as the previous study. Both have to hold in every city.
NIGHTS_R2_BAR = 0.10
NIGHTS_MAE_RATIO = 0.90

# A column missing this often is a median in disguise. Dropped before any
# score is computed, from the missing rate only.
MAX_MISSING_FRACTION = 0.40

N_ESTIMATORS = 200
LEARNING_RATE = 0.08
MODEL_RANDOM_STATE = 17
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 17
INTERVAL_MIN_BIN = 40

# Eligible price and nights models, simplest first.
ELIGIBLE_MODELS = (
    "area_median",
    "linear",
    "ridge",
    "random_forest",
    "xgboost",
)

MODEL_NAMES = {
    "area_median": "Local median rate",
    "linear": "Linear regression",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}

HOME_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "accommodates",
    "beds",
    "bath_is_shared",
    "is_private_room",
    "is_shared_room",
    "is_hotel_room",
    "is_house",
    "is_condo",
    "latitude",
    "longitude",
    "distance_to_center_km",
    "distance_to_airport_km",
]

AMENITY_COLUMNS = [
    "amenity_count",
    "has_ac",
    "has_parking",
    "has_washer",
    "has_kitchen",
    "has_pool",
    "has_hottub",
    "has_waterfront",
]

REVIEW_COLUMNS = [
    "review_score",
    "log_reviews",
    "is_superhost",
]

PRICE_LADDER = (
    ("home", HOME_COLUMNS),
    ("home_and_amenities", HOME_COLUMNS + AMENITY_COLUMNS),
    ("home_amenities_reviews", HOME_COLUMNS + AMENITY_COLUMNS + REVIEW_COLUMNS),
    (
        "home_amenities_reviews_census",
        HOME_COLUMNS + AMENITY_COLUMNS + REVIEW_COLUMNS + list(CENSUS_COLUMNS),
    ),
)

NIGHTS_LADDER = (
    ("home", HOME_COLUMNS),
    ("home_and_amenities", HOME_COLUMNS + AMENITY_COLUMNS),
    (
        "home_amenities_census",
        HOME_COLUMNS + AMENITY_COLUMNS + list(CENSUS_COLUMNS),
    ),
)

# The nights label is built from last-year reviews and the minimum stay.
# Those columns, the price, and availability are not nights features.
NIGHTS_FORBIDDEN = frozenset(
    {
        "price",
        "price_clean",
        "booked_nights",
        "occupancy_rate",
        "number_of_reviews",
        "number_of_reviews_ltm",
        "log_reviews",
        "reviews_per_month",
        "review_score",
        "is_superhost",
        "minimum_nights",
        "minimum_nights_capped",
        "availability_365",
        "has_availability",
        "review_cleanliness",
        "review_location",
        "review_accuracy",
        "review_checkin",
        "review_communication",
    }
)

# The price model may use review history. It may not use the nights label,
# availability, or the price itself.
PRICE_FORBIDDEN = frozenset(
    {
        "price",
        "price_clean",
        "booked_nights",
        "occupancy_rate",
        "number_of_reviews_ltm",
        "availability_365",
        "has_availability",
    }
)


def assert_seeds() -> None:
    if PRIMARY_SEED in FORBIDDEN_SEEDS:
        raise RuntimeError(f"primary seed {PRIMARY_SEED} was already used")
    if set(CONFIRM_SEEDS) & FORBIDDEN_SEEDS:
        raise RuntimeError("a confirmation seed was already used")
    if PRIMARY_SEED in CONFIRM_SEEDS:
        raise RuntimeError("the primary seed is also a confirmation seed")
    if len(set(CONFIRM_SEEDS)) != len(CONFIRM_SEEDS):
        raise RuntimeError("confirmation seeds are not unique")


def choose_simplest(scores: list[tuple[str, float]], margin: float = MODEL_MARGIN) -> str:
    """`scores` is simplest-first. The first entry within `margin` of the best wins."""
    if not scores:
        raise ValueError("no scores")
    best = min(mae for _, mae in scores)
    limit = best * (1.0 + margin)
    for name, mae in scores:
        if mae <= limit + 1e-9:
            return name
    raise RuntimeError("no model fell inside the margin")


def choose_params(trials: list[dict], margin: float = PARAM_MARGIN) -> dict:
    """Keep the simplest trial within `margin` of the best CV MAE.

    Each trial has `mae`, `simplicity` (a tuple; lower is simpler), and `params`.
    """
    if not trials:
        raise ValueError("no trials")
    best = min(float(trial["mae"]) for trial in trials)
    limit = best * (1.0 + margin)
    eligible = [trial for trial in trials if float(trial["mae"]) <= limit + 1e-9]
    eligible.sort(key=lambda trial: (tuple(trial["simplicity"]), float(trial["mae"])))
    return eligible[0]


def choose_nights_target(raw_mae: float, log_mae: float, margin: float = FEATURE_MARGIN) -> str:
    """Direct nights, unless the log target is better by `margin` on CV MAE."""
    if log_mae <= raw_mae * (1.0 - margin):
        return "log"
    return "raw"


def nights_cleared(mae: float, baseline_mae: float, r2: float) -> bool:
    """True only when both halves of the nights bar hold."""
    if baseline_mae <= 0:
        return False
    return float(mae) <= NIGHTS_MAE_RATIO * float(baseline_mae) and float(r2) >= NIGHTS_R2_BAR


def selection_reason(scores: list[tuple[str, float]], chosen: str) -> str:
    """One sentence a person can read. `scores` is simplest-first."""
    named = [(MODEL_NAMES.get(name, name), mae) for name, mae in scores]
    chosen_label = MODEL_NAMES.get(chosen, chosen)
    best_label, best = min(named, key=lambda item: item[1])
    limit = best * (1.0 + MODEL_MARGIN)
    chosen_mae = dict(named)[chosen_label]
    if chosen_label == best_label:
        return (
            f"{chosen_label} had the lowest average error, ${best:,.2f} a night. "
            f"No simpler model was within 2% of that. The cutoff was ${limit:,.2f}."
        )
    return (
        f"The lowest average error was ${best:,.2f} a night, from {best_label}. "
        f"{chosen_label} was ${chosen_mae:,.2f}, inside the 2% band "
        f"(up to ${limit:,.2f}), and it is the simplest model in that band."
    )


def nights_selection_reason(scores: list[tuple[str, float]], chosen: str) -> str:
    """Same 2% rule as the nightly rate, worded in nights rather than dollars."""
    named = [(MODEL_NAMES.get(name, name), mae) for name, mae in scores]
    chosen_label = MODEL_NAMES.get(chosen, chosen)
    best_label, best = min(named, key=lambda item: item[1])
    limit = best * (1.0 + MODEL_MARGIN)
    chosen_mae = dict(named)[chosen_label]
    if chosen_label == best_label:
        return (
            f"{chosen_label} had the lowest average error, {best:.1f} nights. "
            f"No simpler model was within 2% of that. The cutoff was {limit:.1f} nights."
        )
    return (
        f"The lowest average error was {best:.1f} nights, from {best_label}. "
        f"{chosen_label} was {chosen_mae:.1f} nights, inside the 2% band "
        f"(up to {limit:.1f} nights), and it is the simplest model in that band."
    )


def nights_reason(cleared: bool) -> str:
    if cleared:
        return (
            "The nights model cleared the pre-set bar in every city: the average miss "
            "is at least 10% lower than the local median nights, and R² is at least 0.10."
        )
    return (
        "The nights model did not clear the pre-set bar in every city. The bar was an "
        "average miss at least 10% lower than the local median nights, and an R² of at "
        "least 0.10. The app labels nights as a rough estimate."
    )
