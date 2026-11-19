"""Occupancy study design, fixed before the occupancy holdout is scored.

The price study already used seeds 0–4, 7, 11, 23, and 42. Seed 31 is none
of those. It was not searched. The listing ids are written from coordinates
and this seed before any occupancy model is fit.

The target is an estimate, not a booking count. The features are things a
person can know about a house. The columns that enter the estimate
(review counts, minimum nights, availability, price) are not features.
"""

from __future__ import annotations

from features.columns import (
    CENSUS_COLUMNS,
    LOCATION_COLUMNS,
    SIZE_COLUMNS,
    STRUCTURE_COLUMNS,
)
from features.frame import AMENITY_FLAGS, EXTRA_AMENITY_FLAGS

# Not 0, 1, 2, 3, 4, 7, 11, 23, or 42. Chosen before the fit. Not searched.
OCCUPANCY_SEED = 31
OCCUPANCY_GROUP_FRACTION = 0.20
INNER_SEEDS = (0, 1, 2, 3, 4)
INNER_GROUP_FRACTION = 0.20

# Published price-model depth. Not re-opened on this label.
OCCUPANCY_DEPTH = 4

# A city clears the regression bar only when both are true on the holdout.
# 10% is the smallest cut in average error this study will call meaningful.
# R² of 0.10 is the smallest share of variance it will call a fit. The price
# study's 1% rule is for choosing among similar feature sets, not for this.
MAE_RATIO_BAR = 0.90
R2_BAR = 0.10

# Classification bar, also on the holdout, also fixed first. Accuracy has to
# beat the neighbourhood-majority class by 8 points, and balanced accuracy
# has to beat that baseline's balanced accuracy. Tertiles are the classes.
CLASS_ACCURACY_MARGIN = 0.08

# Grade cuts of estimated annual revenue inside the city. Training rows only.
# A is the top fifth, F is the bottom fifth. Not fit to a test metric.
GRADE_QUANTILES = (0.20, 0.40, 0.60, 0.80)
GRADE_ORDER = (
    ("A", "Strong", "p80"),
    ("B", "Above average", "p60"),
    ("C", "Typical", "p40"),
    ("D", "Below average", "p20"),
    ("F", "Weak", None),
)

AMENITY_COLUMNS = [
    "beds",
    "amenity_count",
    *AMENITY_FLAGS,
    *EXTRA_AMENITY_FLAGS,
]

# Anything that builds the nights label, or that a house on the market does
# not have. Checked in tests and again before a model is fit.
FORBIDDEN_COLUMNS = frozenset(
    {
        "price",
        "price_clean",
        "occupancy_rate",
        "booked_nights",
        "number_of_reviews",
        "number_of_reviews_ltm",
        "number_of_reviews_l30d",
        "reviews_per_month",
        "log_reviews",
        "review_score",
        "review_scores_rating",
        "review_cleanliness",
        "review_location",
        "review_accuracy",
        "review_checkin",
        "review_communication",
        "review_scores_value",
        "minimum_nights",
        "minimum_nights_capped",
        "availability_30",
        "availability_60",
        "availability_90",
        "availability_365",
        "is_superhost",
        "log_host_listings",
        "instant_bookable",
        "host_response_rate",
        "host_acceptance_rate",
        "host_response_time_ord",
        "host_identity_verified",
        "log_host_tenure",
        "estimated_revenue",
        "estimated_occupancy",
    }
)


def _prefix(*groups: list[str]) -> list[str]:
    columns: list[str] = []
    for group in groups:
        columns.extend(group)
    return columns


# Simpler prefixes first. The earliest one within 1% of the best development
# MAE is kept. Amenities are last because they describe a listing the host
# has already written, and the app can leave them blank.
FEATURE_LADDER: list[tuple[str, list[str]]] = [
    ("size_and_room", list(SIZE_COLUMNS)),
    ("plus_location", _prefix(SIZE_COLUMNS, LOCATION_COLUMNS)),
    ("plus_census", _prefix(SIZE_COLUMNS, LOCATION_COLUMNS, CENSUS_COLUMNS)),
    (
        "plus_structure",
        _prefix(SIZE_COLUMNS, LOCATION_COLUMNS, CENSUS_COLUMNS, STRUCTURE_COLUMNS),
    ),
    (
        "plus_amenities",
        _prefix(SIZE_COLUMNS, LOCATION_COLUMNS, CENSUS_COLUMNS, STRUCTURE_COLUMNS, AMENITY_COLUMNS),
    ),
]


def ladder_columns(name: str) -> list[str]:
    for candidate, columns in FEATURE_LADDER:
        if candidate == name:
            return list(columns)
    raise KeyError(name)


def assert_columns_allowed(columns: list[str]) -> None:
    banned = [column for column in columns if column in FORBIDDEN_COLUMNS]
    if banned:
        raise RuntimeError(f"Occupancy features include label inputs: {banned}")


def cleared_regression(mae: float, baseline_mae: float, r2: float) -> bool:
    """Holdout bar. Both the error cut and the R² floor are required."""
    if baseline_mae <= 0:
        return False
    return float(mae) <= float(baseline_mae) * MAE_RATIO_BAR and float(r2) >= R2_BAR


def cleared_classification(
    accuracy: float,
    baseline_accuracy: float,
    balanced: float,
    baseline_balanced: float,
) -> bool:
    return float(accuracy) >= float(baseline_accuracy) + CLASS_ACCURACY_MARGIN and float(
        balanced
    ) > float(baseline_balanced)
