"""Column groups shared by the protocol, the feature code, and the model.

Names live here so a feature cannot be added to the model and forgotten in
the write-up, or the other way around. Nothing in this module reads a price.
"""

from __future__ import annotations

# Size of the place and which product is being rented. An entire home is the
# baseline (all three room flags off), which is also how the median baseline
# is keyed, so the model is not credited for a split a median already makes.
SIZE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "accommodates",
    "is_private_room",
    "is_shared_room",
    "is_hotel_room",
]

# Coordinates plus two distances that do not use anyone's price. Latitude and
# longitude let a tree draw a surface; the two distances give that surface
# an anchor a person can explain.
LOCATION_COLUMNS = [
    "latitude",
    "longitude",
    "distance_to_center_km",
    "distance_to_airport_km",
]

# ACS 5-year tract attributes. They move slowly and they are not a function
# of the advertised rate, so they are fair game on a spatial holdout.
CENSUS_COLUMNS = [
    "census_median_income",
    "census_median_rent",
    "census_median_home_value",
    "census_pop_density",
]

# OpenStreetMap counts and distances. Caps and radii are fixed in
# features/place.py; they are not tuned on a test fold.
OSM_COLUMNS = [
    "osm_dist_food_km",
    "osm_n_food_1km",
    "osm_dist_transit_km",
    "osm_n_transit_1km",
    "osm_dist_park_km",
    "osm_n_park_1km",
    "osm_dist_beach_km",
    "osm_n_beach_1km",
    "osm_dist_attraction_km",
    "osm_n_attraction_1km",
]

# Building type, and interactions a single column cannot say: an extra guest
# does not price like an entire home when the product is a room, and the
# drive from downtown can have a different slope for a house than a condo.
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
STRUCTURE_COLUMNS = PROPERTY_COLUMNS + INTERACTION_COLUMNS

# The current ad: amenities, reviews, host. Known for a listing that already
# exists, unknown for a house that has never been listed.
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

# More of the ad than v2 used: beds, a shared bath, host operations, the
# separate review scores, and a few amenities. review_scores_value is not
# here. Guests score "value" by comparing the stay to the price, so that
# column is a consequence of the thing we are predicting.
DETAIL_COLUMNS = [
    "beds",
    "bath_is_shared",
    "log_host_tenure",
    "host_response_rate",
    "host_acceptance_rate",
    "host_response_time_ord",
    "host_identity_verified",
    "review_cleanliness",
    "review_location",
    "review_accuracy",
    "review_checkin",
    "review_communication",
    "has_fireplace",
    "has_dishwasher",
    "has_self_checkin",
    "has_patio",
    "has_bbq",
    "has_ev_charger",
    "allows_pets",
    "has_elevator",
]

# Fixed phrases in the listing's own title and description. The patterns
# live in features/text.py. They were written down before the fit, not
# mined from a test fold. An empty ad is missing, not a string of zeros.
LEXICON_COLUMNS = [
    "log_text_chars",
    "text_mentions_view",
    "text_mentions_luxury",
    "text_mentions_renovated",
    "text_mentions_new",
    "text_mentions_quiet",
    "text_mentions_walk",
    "text_mentions_downtown",
    "text_mentions_water",
]


def text_component_names(n_components: int) -> list[str]:
    """Fold-fit TF-IDF components. The count is fixed in the protocol."""
    return [f"text_svd_{index:02d}" for index in range(n_components)]


# One dummy per study city, used only if a pooled model is selected.
# Hyphens become underscores so the name is a legal feature name.
CITY_DUMMY_COLUMNS = ["city_austin", "city_san_diego", "city_seattle"]


def city_dummy(city_key: str) -> str:
    return "city_" + city_key.replace("-", "_")


# Fold-fit encodings. These use other listings' prices, so they are attached
# inside the training fold only. They are not columns on the raw frame.
SMOOTH_COLUMNS = ["neigh_log_price", "neigh_bed_log_price"]
KNN_COLUMNS = ["knn_log_price"]
# Gaussian weights on the nearest training listings. Same leakage rule.
KERNEL_COLUMNS = ["kernel_log_price"]
# Median log price of training listings with the same room type and bedroom
# bin, within 15 km. Fit inside the training fold only.
COMP_COLUMNS = ["comp_log_price"]

# Monotone directions. A constraint is an inductive bias, not a claim that
# the world is perfectly monotone: more bedrooms should not lower the
# predicted rate, and a private room should not raise it relative to an
# entire home. Density and distances are left free because a dense tract
# can be a downtown or a cheap corridor, and distance to downtown is not
# the same thing as distance to the coast.
MONOTONE_UP = {
    "bedrooms",
    "bathrooms",
    "accommodates",
    "census_median_income",
    "census_median_rent",
    "census_median_home_value",
    *SMOOTH_COLUMNS,
    *KNN_COLUMNS,
}
MONOTONE_DOWN = {"is_private_room", "is_shared_room"}

PRETTY = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "accommodates": "Guest capacity",
    "is_private_room": "Private room",
    "is_shared_room": "Shared room",
    "is_hotel_room": "Hotel room",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "distance_to_center_km": "Distance to downtown",
    "distance_to_airport_km": "Distance to the airport",
    "census_median_income": "Tract median income",
    "census_median_rent": "Tract median rent",
    "census_median_home_value": "Tract median home value",
    "census_pop_density": "Tract population density",
    "osm_dist_food_km": "Distance to food",
    "osm_n_food_1km": "Food places within 1 km",
    "osm_dist_transit_km": "Distance to transit",
    "osm_n_transit_1km": "Transit stops within 1 km",
    "osm_dist_park_km": "Distance to a park",
    "osm_n_park_1km": "Parks within 1 km",
    "osm_dist_beach_km": "Distance to a beach",
    "osm_n_beach_1km": "Beaches within 1 km",
    "osm_dist_attraction_km": "Distance to an attraction",
    "osm_n_attraction_1km": "Attractions within 1 km",
    "is_house": "House",
    "is_cabin": "Cabin",
    "is_cottage": "Cottage",
    "is_condo": "Condo or apartment",
    "is_townhouse": "Townhouse",
    "is_guest": "Guest unit",
    "accommodates_x_private": "Capacity × private room",
    "accommodates_x_shared": "Capacity × shared room",
    "distance_x_house": "Downtown distance × house",
    "distance_x_condo": "Downtown distance × condo",
    "neigh_log_price": "Neighbourhood price level",
    "neigh_bed_log_price": "Neighbourhood × size price level",
    "knn_log_price": "Nearby training prices",
    "minimum_nights_capped": "Minimum nights",
    "amenity_count": "Amenity count",
    "has_ac": "Air conditioning",
    "has_parking": "Parking",
    "has_washer": "Washer",
    "has_dryer": "Dryer",
    "has_pool": "Pool",
    "has_hottub": "Hot tub",
    "has_gym": "Gym",
    "has_waterfront": "Waterfront",
    "has_kitchen": "Kitchen",
    "has_workspace": "Workspace",
    "review_score": "Review score",
    "log_reviews": "Review count",
    "reviews_per_month": "Reviews per month",
    "is_superhost": "Superhost",
    "log_host_listings": "Host's listing count",
    "instant_bookable": "Instant book",
    "beds": "Beds",
    "bath_is_shared": "Shared bathroom",
    "log_host_tenure": "Host tenure",
    "host_response_rate": "Host response rate",
    "host_acceptance_rate": "Host acceptance rate",
    "host_response_time_ord": "Host response time",
    "host_identity_verified": "Verified host",
    "review_cleanliness": "Cleanliness score",
    "review_location": "Location score",
    "review_accuracy": "Accuracy score",
    "review_checkin": "Check-in score",
    "review_communication": "Communication score",
    "has_fireplace": "Fireplace",
    "has_dishwasher": "Dishwasher",
    "has_self_checkin": "Self check-in",
    "has_patio": "Patio or balcony",
    "has_bbq": "BBQ",
    "has_ev_charger": "EV charger",
    "allows_pets": "Pets allowed",
    "has_elevator": "Elevator",
    "log_text_chars": "Length of the listing text",
    "text_mentions_view": "Text mentions a view",
    "text_mentions_luxury": "Text mentions luxury",
    "text_mentions_renovated": "Text mentions a renovation",
    "text_mentions_new": "Text mentions new construction",
    "text_mentions_quiet": "Text mentions quiet",
    "text_mentions_walk": "Text mentions walking",
    "text_mentions_downtown": "Text mentions downtown",
    "text_mentions_water": "Text mentions water",
    "kernel_log_price": "Smoothed nearby prices",
    "comp_log_price": "Comparable listings",
    "months_before": "Months before the study snapshot",
    "listing_wording": "Wording of the listing",
    "listing_embedding": "Wording of the listing",
    "city_austin": "Austin",
    "city_san_diego": "San Diego",
    "city_seattle": "Seattle",
}


def pretty(name: str) -> str:
    if name.startswith("text_svd_") or name.startswith("text_embed_"):
        # Components are numbered for the chart. The app collapses them
        # into one wording reason because a single component is not a
        # sentence a person can say.
        return f"Listing wording {int(name.rsplit('_', 1)[-1]) + 1}"
    return PRETTY.get(name, name.replace("_", " "))


def embed_component_names(n_components: int) -> list[str]:
    """Fold-fit sentence-embedding components. The count is fixed in the protocol."""
    return [f"text_embed_{index:02d}" for index in range(n_components)]
