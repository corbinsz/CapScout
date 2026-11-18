"""The evaluation design, fixed before the locked test is scored.

What is held out
    One spatial test per city. Cells on a 0.03-degree grid (about 3 km of
    latitude) are assigned from coordinates alone. The held-out listing ids
    are frozen in evaluation/locked_test_ids.json. Seed 7 drew that split.
    It is not seed 42 from the previous study: those scores had already been
    read, so a new feature set is not allowed to chase them. The seed was
    chosen before any Census or OpenStreetMap feature was built. It is not
    a value that was searched.

What else is kept out of training
    A host with a listing in the test also loses their other listings from
    the training side. Training listings within 1 km of a test listing are
    dropped. Both rules shrink training; the counts are reported, not tuned.

How models are compared
    The development listings are split the same way, five times (seeds 0–4).
    Mean validation MAE in dollars, averaged over the three cities, makes
    every choice below. The locked test is scored once, after that, and is
    not an input to any of it.

    1. Feature stage 1 is a cumulative prefix: size, location, census, OSM,
       then building type and interactions. The simplest prefix within 1%
       MAE of the best prefix is kept.
    2. Two price encodings, fit inside the training fold only, are tried on
       top of that prefix: a smoothed neighbourhood level, and the median of
       the 40 nearest training listings. An encoding is kept only if it
       improves MAE by at least 1%. If both do, the simpler one wins a 1% tie.
       The smoothed level is the simpler one. Nearest-neighbour prices were
       the encoding that failed under this buffer in the previous study, so
       they have to earn their place rather than being the default.
    3. Listing quality (reviews, amenities, host) is added only if it
       improves MAE by at least 1%. Those fields describe an ad that already
       exists.
    4. Tree depth 4, 6, then 8. Shallower wins within 1%.
    5. LightGBM or CatBoost replaces XGBoost only if its MAE is under 98%
       of XGBoost's. XGBoost is the default because it was the previous
       study's model, not because it is assumed to be better.
    6. Monotone constraints are kept when they are within 1% of the
       unconstrained fit. The constraint is the smaller hypothesis.
    7. An equal-weight average of the two best unconstrained libraries
       replaces the winner only if it is at least 1% better. The weights
       are not fit on the validation fold, so that comparison is not
       optimistic. A learned stacker would need another split; with three
       cities it would mostly be noise.

A diagnostic prefix, location plus OSM and no census, is scored so the
ablation can show whether the two maps substitute for each other. It is
not eligible to be selected. Selecting it would be a second search.
"""

from __future__ import annotations

from data.cities import STUDY_CITIES
from features.columns import (
    CENSUS_COLUMNS,
    LOCATION_COLUMNS,
    OSM_COLUMNS,
    QUALITY_COLUMNS,
    SIZE_COLUMNS,
    STRUCTURE_COLUMNS,
)

TEST_SEED = 7
PREVIOUS_STUDY_SEED = 42
TEST_GROUP_FRACTION = 0.20
INNER_SEEDS = (0, 1, 2, 3, 4)
INNER_GROUP_FRACTION = 0.20
SPATIAL_STEP_DEGREES = 0.03
BUFFER_KM = 1.0
KNN_K = 40
TARGET_PRIOR = 20.0
MAX_TREES = 400
EARLY_STOPPING_ROUNDS = 30
# Nominal coverage of the prediction interval. Chosen before looking at
# calibration. 80% is wide enough to be useful and narrow enough to check.
INTERVAL_LEVEL = 0.80

# Relative MAE. A simpler feature prefix is kept when it is this close to
# the best prefix. An added group has to clear the same 1% the other way.
# A new library has to clear 2% before it replaces XGBoost.
SIMPLER_WITHIN = 1.01
ADDON_MUST_BEAT = 0.99
LIGHTGBM_MUST_BEAT = 0.98
LIBRARY_MUST_BEAT = LIGHTGBM_MUST_BEAT
BLEND_MUST_BEAT = 0.99

STUDY_RATIONALE = (
    "Austin, San Diego, and Seattle stay the study markets. Each is one metro, "
    "the Inside Airbnb snapshot is from 2025, and bedrooms are almost complete. "
    "The other snapshots on disk were screened again for this round. Hawaii is "
    "several islands in one file. Asheville's bedrooms column is empty. Las Vegas, "
    "Fort Lauderdale, Nashville, and Washington have no nightly price on roughly "
    "20–30% of rows. Boston, Denver, Dallas, Fort Worth, and New Orleans are "
    "missing a price on about 9–20% and are smaller. A fourth market with a "
    "large unpriced slice would not make the spatial test more honest, so the "
    "headline table is still three cities."
)


def _prefix(*groups: list[str]) -> list[str]:
    columns: list[str] = []
    for group in groups:
        columns.extend(group)
    return columns


# Stage 1, in the only order the simplicity rule considers. Each entry adds
# the next source on top of the previous one. spatial mode is "none" until
# a later stage opts into a fold-fit encoding.
STAGE1: list[tuple[str, list[str], str]] = [
    ("size_and_room", _prefix(SIZE_COLUMNS), "none"),
    ("plus_location", _prefix(SIZE_COLUMNS, LOCATION_COLUMNS), "none"),
    ("plus_census", _prefix(SIZE_COLUMNS, LOCATION_COLUMNS, CENSUS_COLUMNS), "none"),
    (
        "plus_osm",
        _prefix(SIZE_COLUMNS, LOCATION_COLUMNS, CENSUS_COLUMNS, OSM_COLUMNS),
        "none",
    ),
    (
        "plus_structure",
        _prefix(SIZE_COLUMNS, LOCATION_COLUMNS, CENSUS_COLUMNS, OSM_COLUMNS, STRUCTURE_COLUMNS),
        "none",
    ),
]

# Not eligible. Answers "does OSM help if census is absent?"
OSM_WITHOUT_CENSUS: tuple[str, list[str], str] = (
    "osm_without_census",
    _prefix(SIZE_COLUMNS, LOCATION_COLUMNS, OSM_COLUMNS),
    "none",
)

FEATURE_SET_WHY = {
    "size_and_room": "How big the place is, and whether the guest gets the whole home or a room.",
    "plus_location": "Where it sits: coordinates, distance to downtown, and distance to the airport.",
    "plus_census": (
        "The census tract around it: median income, median rent, median home value, "
        "and population density. A 2024 ACS 5-year join, not a listing's own price."
    ),
    "plus_osm": (
        "What is nearby in OpenStreetMap: food, transit, parks, beaches, and attractions. "
        "Counts within 1 km and distance to the nearest, capped at 20 km."
    ),
    "plus_structure": (
        "Building type, and two interactions: capacity priced differently for a room, "
        "and downtown distance priced differently for a house than a condo."
    ),
    "osm_without_census": "OpenStreetMap on top of location, with census left out, as a substitution check.",
    "target_smooth": (
        "Smoothed mean log price of the neighbourhood, and of neighbourhood by bedroom bin, "
        "fit on the training fold only. Unseen neighbourhoods fall back to the training mean."
    ),
    "target_knn": (
        "Median log price of the 40 nearest other training listings. Under a 1 km buffer "
        "those neighbours are not next door."
    ),
    "listing_quality": (
        "Amenities, reviews, and host fields from the current ad. Available for a listing "
        "that exists, not for a house that has never been listed."
    ),
}

# Kept so older notes that say "FEATURE_SETS" still point at the eligible prefixes.
FEATURE_SETS = STAGE1

DEPTHS = (4, 6, 8)
LIBRARIES = ("xgboost", "lightgbm", "catboost")


def study_cities() -> tuple[str, ...]:
    return STUDY_CITIES


def stage1_spec(name: str) -> tuple[list[str], str]:
    for candidate, columns, mode in STAGE1:
        if candidate == name:
            return list(columns), mode
    raise KeyError(name)


def resolve_features(stage1: str, addon: str | None, quality: bool) -> tuple[list[str], str]:
    """Static columns and the fold-fit encoding mode for a chosen plan."""
    columns, _mode = stage1_spec(stage1)
    spatial = "none"
    if addon == "target_smooth":
        spatial = "smooth"
    elif addon == "target_knn":
        spatial = "knn"
    elif addon:
        raise ValueError(addon)
    if quality:
        columns = list(columns) + list(QUALITY_COLUMNS)
    return columns, spatial
