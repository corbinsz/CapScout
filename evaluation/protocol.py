"""The evaluation design, fixed before any test metric is computed.

What is held out
    One spatial test per city. Cells on a 0.03-degree grid (about 3 km of
    latitude) are assigned from coordinates alone. GroupShuffleSplit with
    seed 42 holds out 20% of those cells. That test is not used to pick a
    feature set, a tree depth, or XGBoost versus LightGBM.

What else is kept out of training
    A host with a listing in the test also loses their other listings from
    the training side, so a host's pricing habits cannot cross the split.
    Training listings within 1 km of a test listing are dropped as well.
    Otherwise a cell-edge home would still see its neighbor. Both rules
    shrink training; the counts are reported, not tuned.

How models are compared
    The remaining development listings are split the same way, five times
    (seeds 0–4). Mean validation MAE across those seeds, averaged over the
    three cities, picks the feature prefix, then the tree depth, then the
    library. A simpler prefix wins if it is within 1% MAE of the best one.
    LightGBM replaces XGBoost only if it is at least 2% better. Those
    thresholds are here so a tiny difference cannot change the story.

The locked test is scored once, after those choices, by a model refit on
the development listings.
"""

from __future__ import annotations

from data.cities import STUDY_CITIES

TEST_SEED = 42
TEST_GROUP_FRACTION = 0.20
INNER_SEEDS = (0, 1, 2, 3, 4)
INNER_GROUP_FRACTION = 0.20
SPATIAL_STEP_DEGREES = 0.03
BUFFER_KM = 1.0
KNN_K = 40
TARGET_PRIOR = 20.0
MAX_TREES = 400
EARLY_STOPPING_ROUNDS = 30

# Relative MAE. A simpler feature prefix is kept when it is this close to
# the best prefix. LightGBM has to clear a larger gap before it replaces
# XGBoost, which is the pre-specified default.
SIMPLER_WITHIN = 1.01
LIGHTGBM_MUST_BEAT = 0.98

STUDY_RATIONALE = (
    "Austin, San Diego, and Seattle are the study markets. Each is one metro "
    "(latitude standard deviation about 0.05–0.07, not an archipelago), the "
    "Inside Airbnb snapshot is from 2025, and bedrooms are almost complete. "
    "Austin has 10,323 listings inside the $25–$2,000 window and only 16 "
    "missing prices. San Diego has 11,613 and Seattle 6,004; Seattle has "
    "almost no listings above the $2,000 cap, so the tail does not dominate. "
    "Hawaii is many islands in one file. Las Vegas, Fort Lauderdale, and "
    "Nashville are missing a nightly price on roughly 30% of rows. Asheville's "
    "bedrooms column is essentially empty. The other snapshots stay on disk "
    "and are not in the headline results."
)

# Feature prefixes, in the only order the ablation considers. Each entry is
# (name, extra columns beyond the previous prefix, whether fold-fit spatial
# encodings are included). Columns are cumulative.
SIZE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "accommodates",
    "is_private_room",
    "is_shared_room",
    "is_hotel_room",
]
LOCATION_COLUMNS = [
    "latitude",
    "longitude",
    "distance_to_center_km",
    "distance_to_airport_km",
]
PROPERTY_COLUMNS = [
    "is_house",
    "is_cabin",
    "is_cottage",
    "is_condo",
    "is_townhouse",
    "is_guest",
]
INTERACTION_COLUMNS = [
    "accommodates_x_private",
    "accommodates_x_shared",
    "distance_x_house",
    "distance_x_condo",
]
QUALITY_COLUMNS = [
    "minimum_nights_capped",
    "amenity_count",
    "has_ac",
    "has_parking",
    "has_washer",
    "has_dryer",
    "has_pool",
    "has_hottub",
    "has_gym",
    "has_waterfront",
    "has_kitchen",
    "has_workspace",
    "review_score",
    "log_reviews",
    "reviews_per_month",
    "is_superhost",
    "log_host_listings",
    "instant_bookable",
]


# (name, columns, use_spatial_encodings)
def _prefix(*groups: list[str]) -> list[str]:
    columns: list[str] = []
    for group in groups:
        columns.extend(group)
    return columns


FEATURE_SETS: list[tuple[str, list[str], bool]] = [
    ("size_and_room", _prefix(SIZE_COLUMNS), False),
    ("plus_location", _prefix(SIZE_COLUMNS, LOCATION_COLUMNS), False),
    ("plus_property_type", _prefix(SIZE_COLUMNS, LOCATION_COLUMNS, PROPERTY_COLUMNS), False),
    (
        "plus_interactions",
        _prefix(SIZE_COLUMNS, LOCATION_COLUMNS, PROPERTY_COLUMNS, INTERACTION_COLUMNS),
        True,
    ),
    (
        "plus_listing_quality",
        _prefix(
            SIZE_COLUMNS,
            LOCATION_COLUMNS,
            PROPERTY_COLUMNS,
            INTERACTION_COLUMNS,
            QUALITY_COLUMNS,
        ),
        True,
    ),
]

FEATURE_SET_WHY = {
    "size_and_room": "How big the place is and whether it is a whole home, a room, or a shared room.",
    "plus_location": "Where it sits: coordinates, distance to the city center, and distance to the airport.",
    "plus_property_type": "What kind of building: house, condo, cabin, cottage, townhouse, or guest unit.",
    "plus_interactions": (
        "Interactions a single column cannot express: capacity priced differently for a private room, "
        "distance priced differently for a house than a condo, neighbourhood and neighbourhood-by-bedroom "
        "log-price levels, and the median log price of the 40 nearest other training listings."
    ),
    "plus_listing_quality": (
        "How the current listing is equipped and reviewed: amenities, minimum nights, review score and "
        "volume, superhost, host portfolio size, instant book. These describe an existing ad, not a house "
        "that has never been listed."
    ),
}


def study_cities() -> tuple[str, ...]:
    return STUDY_CITIES
