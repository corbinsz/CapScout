"""Census tract attributes and OpenStreetMap distances.

Both are functions of where a listing sits, not of what it charges. The
tract table and the POI extract are built by `python -m data.external`
and cached under data/external/<city>/. This module only joins them.

Why a point-in-polygon join, not a ZIP code: Inside Airbnb does not
publish a ZIP, and ZIP boundaries are a poor stand-in for a tract. The
listing coordinate is jittered by about 150 meters, which is smaller than
a tract, so the tract is the right grain and the jitter is the error bar.

Why distance to a park uses the park's geometry nodes: a large park's
center can sit a kilometer from a listing that is on the park's edge.
Restaurants and stops are small, so their center is the place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import shape

from features.columns import CENSUS_COLUMNS, OSM_COLUMNS

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = REPO / "data" / "external"

# A missing POI is not "infinitely far." Twenty kilometers is past the point
# where another restaurant changes a nightly rate, and it keeps a city with
# no beach (Austin) from feeding the model a huge meaningless number.
DISTANCE_CAP_KM = 20.0
COUNT_RADIUS_KM = 1.0
# Airbnb jitters coordinates by ~150 m, and a listing can sit just offshore
# of the tract that contains the building. Two kilometers is a rescue for
# that miss, not a licence to borrow a far-away tract's income.
NEAREST_TRACT_KM = 2.0
EARTH_KM = 6371.0

# POI categories whose distance is measured to geometry nodes (areas),
# versus a single center point (places).
AREA_CATEGORIES = {"park", "beach"}

OSM_SPEC = [
    ("food", "osm_dist_food_km", "osm_n_food_1km"),
    ("transit", "osm_dist_transit_km", "osm_n_transit_1km"),
    ("park", "osm_dist_park_km", "osm_n_park_1km"),
    ("beach", "osm_dist_beach_km", "osm_n_beach_1km"),
    ("attraction", "osm_dist_attraction_km", "osm_n_attraction_1km"),
]


def project_km(lat: np.ndarray, lon: np.ndarray, lat0: float) -> np.ndarray:
    """Local equirectangular projection in kilometers.

    Accurate to well under 50 meters inside one metro, which is tighter
    than the coordinate jitter on the listings. A full haversine tree
    would be slower and would not change a 1 km count.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = np.radians(lon) * np.cos(np.radians(lat0)) * EARTH_KM
    y = np.radians(lat) * EARTH_KM
    return np.column_stack([x, y])


def nearest_and_count(
    lat: np.ndarray,
    lon: np.ndarray,
    poi_lat: np.ndarray,
    poi_lon: np.ndarray,
    *,
    radius_km: float = COUNT_RADIUS_KM,
    cap_km: float = DISTANCE_CAP_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the nearest POI, and how many sit inside radius_km.

    An empty POI extract returns the cap and a zero count. That is a real
    statement ("nothing in the extract"), not a missing value the model
    would median-fill into a neighbourhood median.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = len(lat)
    distances = np.full(n, cap_km, dtype=float)
    counts = np.zeros(n, dtype=float)
    if n == 0 or len(poi_lat) == 0:
        return distances, counts
    lat0 = float(np.nanmean(lat))
    tree = cKDTree(project_km(poi_lat, poi_lon, lat0))
    query = project_km(lat, lon, lat0)
    dist, _idx = tree.query(query, k=1)
    dist = np.atleast_1d(np.asarray(dist, dtype=float))
    distances = np.minimum(dist, cap_km)
    # query_ball_point returns a Python list per row. The extracts are
    # city-sized, so this stays a few seconds rather than a spatial index
    # we would have to maintain.
    neighbors = tree.query_ball_point(query, r=radius_km)
    counts = np.array([len(hits) for hits in neighbors], dtype=float)
    return distances, counts


def _as_polygons(features: list[dict]):
    geoms = []
    records = []
    for feature in features:
        geom = shape(feature["geometry"])
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        geoms.append(geom)
        records.append(feature["properties"])
    return geoms, records


@dataclass
class PlaceIndex:
    """Tract polygons and POI trees for one city, loaded once."""

    city: str
    geoms: list
    tree: STRtree
    properties: list[dict]
    centroids_lat: np.ndarray
    centroids_lon: np.ndarray
    medians: dict[str, float]
    pois: dict[str, tuple[np.ndarray, np.ndarray]]
    bbox: tuple[float, float, float, float]

    def census_for(self, lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        n = len(lat)
        assigned = np.full(n, -1, dtype=int)
        # 0 = point-in-polygon, 1 = nearest centroid within 2 km, 2 = city median.
        join = np.full(n, 2, dtype=np.int8)
        if n and len(self.geoms):
            points = _points(lon, lat)
            pairs = np.asarray(self.tree.query(points, predicate="intersects"))
            # An array of points returns shape (2, n_hits): row 0 is the point,
            # row 1 the polygon. An empty hit set is (2, 0), so `len` is the
            # wrong check. The first hit wins; overlapping edges are rare and
            # either tract is a fair description.
            if pairs.ndim == 2 and pairs.shape[1] > 0:
                point_idx = np.asarray(pairs[0], dtype=int)
                poly_idx = np.asarray(pairs[1], dtype=int)
                order = np.argsort(point_idx, kind="mergesort")
                point_idx = point_idx[order]
                poly_idx = poly_idx[order]
                first = np.ones(len(point_idx), dtype=bool)
                first[1:] = point_idx[1:] != point_idx[:-1]
                assigned[point_idx[first]] = poly_idx[first]
                join[point_idx[first]] = 0

            missing = np.flatnonzero(assigned < 0)
            if len(missing) and len(self.centroids_lat):
                lat0 = float(np.nanmean(lat))
                tree = cKDTree(project_km(self.centroids_lat, self.centroids_lon, lat0))
                dist, idx = tree.query(project_km(lat[missing], lon[missing], lat0), k=1)
                close = np.atleast_1d(dist) <= NEAREST_TRACT_KM
                assigned[missing[close]] = np.atleast_1d(idx)[close]
                join[missing[close]] = 1

        columns = {name: np.empty(n, dtype=float) for name in CENSUS_COLUMNS}
        imputed = np.zeros(n, dtype=bool)
        for i in range(n):
            tract = assigned[i]
            if tract < 0:
                imputed[i] = True
                for name in CENSUS_COLUMNS:
                    columns[name][i] = self.medians[name]
                continue
            props = self.properties[tract]
            for name in CENSUS_COLUMNS:
                value = props.get(name)
                if value is None or not np.isfinite(value):
                    columns[name][i] = self.medians[name]
                    imputed[i] = True
                else:
                    columns[name][i] = float(value)
        frame = pd.DataFrame(columns)
        frame["census_imputed"] = imputed
        frame["census_join"] = join
        return frame

    def osm_for(self, lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        data = {}
        for category, dist_name, count_name in OSM_SPEC:
            poi_lat, poi_lon = self.pois.get(category, (np.array([]), np.array([])))
            dist, count = nearest_and_count(lat, lon, poi_lat, poi_lon)
            data[dist_name] = dist
            data[count_name] = count
        return pd.DataFrame(data)

    def covers(self, lat: float, lon: float) -> bool:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _points(lon: np.ndarray, lat: np.ndarray):
    from shapely import points

    return points(np.column_stack([np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)]))


def context_dir(city: str) -> Path:
    return EXTERNAL / city


def context_ready(city: str) -> bool:
    directory = context_dir(city)
    return (directory / "tracts.geojson").is_file() and (directory / "pois.csv.gz").is_file()


@lru_cache(maxsize=8)
def get_index(city: str) -> PlaceIndex:
    """Load one city's extract. Cached for the life of the process."""
    directory = context_dir(city)
    if not context_ready(city):
        raise FileNotFoundError(
            f"No Census/OSM extract for {city}. Build it with: python -m data.external"
        )
    payload = json.loads((directory / "tracts.geojson").read_text())
    meta = json.loads((directory / "context.json").read_text())
    geoms, properties = _as_polygons(payload["features"])
    if not geoms:
        raise RuntimeError(f"{city} tract extract has no polygons")
    centroids = np.array([geom.centroid.coords[0] for geom in geoms], dtype=float)
    pois_frame = pd.read_csv(directory / "pois.csv.gz")
    pois: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for category, part in pois_frame.groupby("category"):
        pois[str(category)] = (
            part["latitude"].to_numpy(dtype=float),
            part["longitude"].to_numpy(dtype=float),
        )
    medians = {name: float(meta["census_medians"][name]) for name in CENSUS_COLUMNS}
    bbox = tuple(float(v) for v in meta["bbox"])
    return PlaceIndex(
        city=city,
        geoms=geoms,
        tree=STRtree(geoms),
        properties=properties,
        centroids_lon=centroids[:, 0],
        centroids_lat=centroids[:, 1],
        medians=medians,
        pois=pois,
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
    )


def attach_place_features(frame: pd.DataFrame, city: str) -> tuple[pd.DataFrame, dict]:
    """Add census and OSM columns. Does not read the price column."""
    index = get_index(city)
    lat = frame["latitude"].to_numpy(dtype=float)
    lon = frame["longitude"].to_numpy(dtype=float)
    census = index.census_for(lat, lon)
    osm = index.osm_for(lat, lon)
    out = frame.copy()
    for column in CENSUS_COLUMNS:
        out[column] = census[column].to_numpy()
    for column in OSM_COLUMNS:
        out[column] = osm[column].to_numpy()
    join = census["census_join"].to_numpy()
    info = {
        # A suppressed ACS estimate is filled with the city tract median.
        # That is not the same thing as missing the polygon.
        "census_field_filled_rows": int(census["census_imputed"].sum()),
        "census_field_filled_fraction": (
            float(census["census_imputed"].mean()) if len(census) else 0.0
        ),
        "census_polygon_fraction": float(np.mean(join == 0)) if len(join) else 0.0,
        "census_centroid_fallback_fraction": float(np.mean(join == 1)) if len(join) else 0.0,
        "census_city_median_fraction": float(np.mean(join == 2)) if len(join) else 0.0,
        "outside_poi_bbox": int(
            (
                ~(
                    (lat >= index.bbox[1])
                    & (lat <= index.bbox[3])
                    & (lon >= index.bbox[0])
                    & (lon <= index.bbox[2])
                )
            ).sum()
        ),
    }
    return out, info


def features_at(city: str, lat: float, lon: float) -> dict[str, float]:
    """One coordinate, for the local app. Same join as the training rows."""
    index = get_index(city)
    census = index.census_for(np.array([lat]), np.array([lon]))
    osm = index.osm_for(np.array([lat]), np.array([lon]))
    row = {column: float(census.iloc[0][column]) for column in CENSUS_COLUMNS}
    row.update({column: float(osm.iloc[0][column]) for column in OSM_COLUMNS})
    row["census_imputed"] = bool(census.iloc[0]["census_imputed"])
    row["census_join"] = int(census.iloc[0]["census_join"])
    row["inside_extract"] = bool(index.covers(lat, lon))
    return row
