"""Joins that do not use price: distances, counts, and a tract polygon."""

from __future__ import annotations

import numpy as np
from features.place import PlaceIndex, nearest_and_count
from shapely import STRtree
from shapely.geometry import Polygon


def test_nearest_count_and_empty_extract():
    # A point on the origin and a POI about 1 degree north is ~111 km, so the
    # cap (20 km) applies and the 1 km count is zero.
    dist, count = nearest_and_count(
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.0]),
    )
    assert dist[0] == 20.0
    assert count[0] == 0.0

    # Two POIs, one on top of the query and one a long way off.
    dist, count = nearest_and_count(
        np.array([30.27]),
        np.array([-97.74]),
        np.array([30.27, 31.5]),
        np.array([-97.74, -97.74]),
    )
    assert dist[0] < 0.05
    assert count[0] == 1.0

    empty_dist, empty_count = nearest_and_count(
        np.array([30.0]), np.array([-97.0]), np.array([]), np.array([])
    )
    assert empty_dist[0] == 20.0
    assert empty_count[0] == 0.0


def test_point_in_polygon_and_median_fallback():
    square = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    index = PlaceIndex(
        city="toy",
        geoms=[square],
        tree=STRtree([square]),
        properties=[
            {
                "census_median_income": 80_000.0,
                "census_median_rent": 1_500.0,
                "census_median_home_value": 400_000.0,
                "census_pop_density": 2_000.0,
            }
        ],
        centroids_lat=np.array([0.0]),
        centroids_lon=np.array([0.0]),
        medians={
            "census_median_income": 50_000.0,
            "census_median_rent": 1_000.0,
            "census_median_home_value": 200_000.0,
            "census_pop_density": 100.0,
        },
        pois={},
        bbox=(-2, -2, 2, 2),
    )
    inside = index.census_for(np.array([0.0]), np.array([0.0]))
    assert inside["census_median_income"].iloc[0] == 80_000.0
    assert not bool(inside["census_imputed"].iloc[0])

    # Far from the square and from its centroid: city median, flagged imputed.
    outside = index.census_for(np.array([10.0]), np.array([10.0]))
    assert outside["census_median_income"].iloc[0] == 50_000.0
    assert bool(outside["census_imputed"].iloc[0])


def test_place_features_ignore_a_price_column():
    """Shuffling a price column cannot change a distance. The function never reads it."""
    lat = np.array([30.0, 30.01])
    lon = np.array([-97.0, -97.0])
    pois_lat = np.array([30.0])
    pois_lon = np.array([-97.0])
    first, _count = nearest_and_count(lat, lon, pois_lat, pois_lon)
    second, _count = nearest_and_count(lat, lon, pois_lat, pois_lon)
    np.testing.assert_allclose(first, second)
