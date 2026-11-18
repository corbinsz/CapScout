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

# Fold-fit encodings. These use other listings' prices, so they are attached
# inside the training fold only. They are not columns on the raw frame.
SMOOTH_COLUMNS = ["neigh_log_price", "neigh_bed_log_price"]
KNN_COLUMNS = ["knn_log_price"]

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
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))
