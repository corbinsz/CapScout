"""Download the Census and OpenStreetMap extracts the study is allowed to use.

Census ACS 5-year tables and TIGER/Line tracts are U.S. government works,
which are public domain. The Census API now requires a key, so the tables
are read from the Census summary files on www2.census.gov instead. That is
the same published estimate, and it can be downloaded without an account.

OpenStreetMap is ODbL. Overpass is the automated-download path: this module
sends a bounded query, keeps coordinates and a category, and drops names.
The cached points are a filtered excerpt. Credit OpenStreetMap contributors
and keep the ODbL notice in DATA.md with any redistribution.

Nothing here reads a listing price. The bbox comes from coordinates only,
padded so a park or beach just outside the listings is still in the extract.
"""

from __future__ import annotations

import json
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapefile
from features.columns import CENSUS_COLUMNS
from shapely.geometry import box, mapping, shape

from data.cities import CITIES
from data.ingest import listings_path

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = REPO / "data" / "external"
CACHE = EXTERNAL / "_cache"

ACS_YEAR = 2024
TIGER_YEAR = 2024
BBOX_PAD_DEGREES = 0.12

# ACS table id -> the estimate column in the table-based summary file.
# E001 is the estimate. The margin of error is the next field and is not used:
# a model of nightly rates does not need the sampling error as a feature, and
# adding it would be a second, collinear column.
ACS_TABLES = {
    "B19013": "census_median_income",
    "B25064": "census_median_rent",
    "B25077": "census_median_home_value",
    "B01003": "census_population",
}

STATE_FIPS = {
    "TX": "48",
    "CA": "06",
    "WA": "53",
    "CO": "08",
    "TN": "47",
    "LA": "22",
    "DC": "11",
    "MA": "25",
    "NV": "32",
    "FL": "12",
    "NC": "37",
    "HI": "15",
}

OVERPASS_ENDPOINTS = (
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
)
USER_AGENT = "CapScout/1.0 (personal noncommercial study of public OSM data)"

# Area categories keep geometry nodes so distance is to the edge of a park
# or beach, not to a centroid that can sit a kilometer inland. Point
# categories use the object's center, which is the door, not the parcel.
POI_CATEGORIES = {
    "food": {
        "geom": False,
        "filters": [
            '["amenity"="restaurant"]',
            '["amenity"="cafe"]',
            '["amenity"="fast_food"]',
        ],
    },
    "transit": {
        "geom": False,
        "filters": [
            '["highway"="bus_stop"]',
            '["railway"="station"]',
            '["railway"="halt"]',
            '["railway"="tram_stop"]',
            '["station"="subway"]',
        ],
    },
    "park": {
        "geom": True,
        "filters": [
            '["leisure"="park"]',
            '["leisure"="garden"]',
            '["leisure"="nature_reserve"]',
        ],
    },
    "beach": {
        "geom": True,
        "filters": ['["natural"="beach"]'],
    },
    "attraction": {
        "geom": False,
        "filters": [
            '["tourism"="attraction"]',
            '["tourism"="museum"]',
            '["tourism"="viewpoint"]',
        ],
    },
}


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def _download(session: requests.Session, url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  downloading {url}", flush=True)
    with session.get(url, timeout=180, stream=True) as response:
        response.raise_for_status()
        temporary = dest.with_suffix(dest.suffix + ".part")
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    handle.write(chunk)
        temporary.replace(dest)
    return dest


def city_bbox(city: str) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat), padded. Coordinates only."""
    frame = pd.read_csv(listings_path(city), usecols=["latitude", "longitude"], low_memory=False)
    lat = pd.to_numeric(frame["latitude"], errors="coerce")
    lon = pd.to_numeric(frame["longitude"], errors="coerce")
    pad = BBOX_PAD_DEGREES
    return (
        float(lon.min()) - pad,
        float(lat.min()) - pad,
        float(lon.max()) + pad,
        float(lat.max()) + pad,
    )


def _census_number(text: str, *, allow_zero: bool) -> float:
    text = (text or "").strip()
    if not text or text == ".":
        return float("nan")
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    # ACS puts missing estimates in large negative sentinels.
    if value < 0:
        return float("nan")
    if not allow_zero and value == 0:
        return float("nan")
    return value


def _acs_lookup(session: requests.Session, table: str) -> dict[str, float]:
    """11-digit tract GEOID -> estimate, for every US tract in the table."""
    url = (
        f"https://www2.census.gov/programs-surveys/acs/summary_file/{ACS_YEAR}"
        f"/table-based-SF/data/5YRData/acsdt5y{ACS_YEAR}-{table.lower()}.dat"
    )
    path = _download(session, url, CACHE / f"acsdt5y{ACS_YEAR}-{table.lower()}.dat")
    allow_zero = table == "B01003"
    found: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            if not line.startswith("1400000US"):
                continue
            geo, estimate, *_rest = line.split("|")
            found[geo[-11:]] = _census_number(estimate, allow_zero=allow_zero)
    if len(found) < 1000:
        raise RuntimeError(f"ACS table {table} did not look like a tract file ({len(found)} rows)")
    return found


def _tracts_for_state(session: requests.Session, state_abbr: str) -> Path:
    fips = STATE_FIPS[state_abbr]
    url = (
        f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/TRACT/"
        f"tl_{TIGER_YEAR}_{fips}_tract.zip"
    )
    zip_path = _download(session, url, CACHE / f"tl_{TIGER_YEAR}_{fips}_tract.zip")
    dest = CACHE / f"tl_{TIGER_YEAR}_{fips}_tract"
    marker = dest / f"tl_{TIGER_YEAR}_{fips}_tract.shp"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest)
    return marker


def _field_map(reader: shapefile.Reader) -> list[str]:
    # The first field is the deletion flag pyshp keeps in the header only.
    return [field[0] for field in reader.fields[1:]]


def build_tracts(
    session: requests.Session, city: str, bbox: tuple[float, float, float, float]
) -> dict:
    state = CITIES[city]["state"]
    shp_path = _tracts_for_state(session, state)
    tables = {table: _acs_lookup(session, table) for table in ACS_TABLES}
    window = box(*bbox)
    reader = shapefile.Reader(str(shp_path))
    fields = _field_map(reader)
    features = []
    try:
        for shaperec in reader.iterShapeRecords():
            values = dict(zip(fields, list(shaperec.record), strict=True))
            geoid = str(values.get("GEOID") or "")
            if len(geoid) != 11:
                continue
            shape_bbox = shaperec.shape.bbox
            if shape_bbox is None:
                continue
            if (
                shape_bbox[2] < bbox[0]
                or shape_bbox[0] > bbox[2]
                or shape_bbox[3] < bbox[1]
                or shape_bbox[1] > bbox[3]
            ):
                continue
            geom = shape(shaperec.shape.__geo_interface__)
            if geom.is_empty or not geom.intersects(window):
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            # ~15 meters. Keeps the join honest relative to coordinate jitter
            # without storing the full TIGER ring in the repo.
            simplified = geom.simplify(0.00015, preserve_topology=True)
            if not simplified.is_empty:
                geom = simplified
            aland = float(values.get("ALAND") or 0)
            population = tables["B01003"].get(geoid, float("nan"))
            if aland > 0 and np.isfinite(population):
                density = population / (aland / 1_000_000.0)
            else:
                density = float("nan")
            properties = {
                "geoid": geoid,
                "census_median_income": _json_number(tables["B19013"].get(geoid, float("nan"))),
                "census_median_rent": _json_number(tables["B25064"].get(geoid, float("nan"))),
                "census_median_home_value": _json_number(tables["B25077"].get(geoid, float("nan"))),
                "census_pop_density": _json_number(density),
            }
            features.append(
                {"type": "Feature", "geometry": mapping(geom), "properties": properties}
            )
    finally:
        reader.close()

    if len(features) < 10:
        raise RuntimeError(f"{city} tract join kept only {len(features)} tracts")
    matched = sum(
        1 for feature in features if feature["properties"]["census_median_income"] is not None
    )
    if matched < 0.5 * len(features):
        raise RuntimeError(f"{city} ACS join matched only {matched} of {len(features)} tracts")

    medians = {}
    for column in CENSUS_COLUMNS:
        values = [
            feature["properties"][column]
            for feature in features
            if feature["properties"][column] is not None
        ]
        medians[column] = float(np.median(values))
    # Fill is a property of the tract table, not of listing prices, and it is
    # applied at join time so a waterfront point still has a number. The
    # imputed share is reported; the fill is not a price leak.
    directory = EXTERNAL / city
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tracts.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )
    return {"n_tracts": len(features), "acs_matched_income": matched, "census_medians": medians}


def _json_number(value: float):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _overpass(session: requests.Session, query: str) -> dict:
    last_error = "no attempt"
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(3):
            try:
                response = session.post(endpoint, data={"data": query}, timeout=200)
            except requests.RequestException as exc:
                last_error = f"{endpoint}: {exc}"
                time.sleep(3 * (attempt + 1))
                continue
            text = response.text
            if response.status_code == 200 and text.lstrip().startswith("{"):
                payload = response.json()
                remark = str(payload.get("remark") or "")
                # The public servers say "timed out", which does not contain
                # the substring "timeout". An empty element list with a remark
                # is a failure, not a city with no restaurants.
                lowered = remark.lower()
                failed = any(token in lowered for token in ("error", "timeout", "timed out"))
                if failed or (remark and not payload.get("elements")):
                    last_error = f"{endpoint}: {remark[:240] or 'empty elements'}"
                    time.sleep(3 * (attempt + 1))
                    continue
                return payload
            last_error = f"{endpoint}: HTTP {response.status_code} {text[:180]}"
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Overpass failed ({last_error})")


def _query_elements(
    session: requests.Session, bbox, filters: list[str], geom: bool, depth: int = 0
):
    south, west, north, east = bbox
    box_text = f"({south:.5f},{west:.5f},{north:.5f},{east:.5f})"
    clauses = []
    for tag in filters:
        clauses.append(f"node{tag}{box_text};")
        clauses.append(f"way{tag}{box_text};")
    out = "out geom;" if geom else "out center;"
    query = f"[out:json][timeout:180];({''.join(clauses)});{out}"
    try:
        payload = _overpass(session, query)
        elements = payload.get("elements") or []
        # A city-scale query that comes back empty is a timeout or a truncated
        # reply, not evidence. Splitting into quadrants asks the same question
        # in pieces. A quadrant is allowed to be empty (a corner with no beach).
        if not elements and depth == 0:
            raise RuntimeError("empty Overpass result for the full city bbox")
        return elements
    except RuntimeError:
        if depth >= 2:
            raise
        # A city-sized query sometimes times out. Four quadrants are the same
        # question, asked in pieces, not a different sample.
        mid_lat = (south + north) / 2
        mid_lon = (west + east) / 2
        quadrants = [
            (south, west, mid_lat, mid_lon),
            (south, mid_lon, mid_lat, east),
            (mid_lat, west, north, mid_lon),
            (mid_lat, mid_lon, north, east),
        ]
        elements = []
        for quadrant in quadrants:
            elements.extend(_query_elements(session, quadrant, filters, geom, depth + 1))
            time.sleep(1)
        return elements


def _element_points(element: dict, geom: bool) -> list[tuple[float, float]]:
    if geom and element.get("geometry"):
        coords = element["geometry"]
        if len(coords) > 500:
            coords = coords[::2]
        return [(float(point["lat"]), float(point["lon"])) for point in coords if "lat" in point]
    if "lat" in element and "lon" in element:
        return [(float(element["lat"]), float(element["lon"]))]
    center = element.get("center") or {}
    if "lat" in center and "lon" in center:
        return [(float(center["lat"]), float(center["lon"]))]
    return []


def build_pois(
    session: requests.Session, city: str, bbox: tuple[float, float, float, float]
) -> dict:
    directory = EXTERNAL / city
    directory.mkdir(parents=True, exist_ok=True)
    # Overpass wants (south, west, north, east).
    overpass_bbox = (bbox[1], bbox[0], bbox[3], bbox[2])
    counts = {}
    frames = []
    for category, spec in POI_CATEGORIES.items():
        cache_path = directory / f"pois_{category}.csv.gz"
        if cache_path.exists():
            part = pd.read_csv(cache_path)
            print(f"  {city} {category}: {len(part)} cached points", flush=True)
        else:
            print(f"  {city} {category}: querying OpenStreetMap", flush=True)
            elements = _query_elements(session, overpass_bbox, spec["filters"], spec["geom"])
            seen: set[tuple[float, float]] = set()
            rows = []
            for element in elements:
                for lat, lon in _element_points(element, spec["geom"]):
                    key = (round(lat, 5), round(lon, 5))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"category": category, "latitude": key[0], "longitude": key[1]})
            part = pd.DataFrame(rows, columns=["category", "latitude", "longitude"])
            part.to_csv(cache_path, index=False, compression="gzip")
            print(f"  {city} {category}: {len(part)} points", flush=True)
            time.sleep(2)
        counts[category] = int(len(part))
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(directory / "pois.csv.gz", index=False, compression="gzip")
    return counts


def _summarize_tracts(city: str) -> dict:
    """Medians from a tract file that was already written. No prices involved."""
    payload = json.loads((EXTERNAL / city / "tracts.geojson").read_text())
    features = payload["features"]
    medians = {}
    for column in CENSUS_COLUMNS:
        values = [
            feature["properties"][column]
            for feature in features
            if feature["properties"].get(column) is not None
        ]
        medians[column] = float(np.median(values))
    matched = sum(
        1 for feature in features if feature["properties"].get("census_median_income") is not None
    )
    return {"n_tracts": len(features), "acs_matched_income": matched, "census_medians": medians}


def ensure_city(
    city: str, *, refresh: bool = False, session: requests.Session | None = None
) -> dict:
    directory = EXTERNAL / city
    context_path = directory / "context.json"
    ready = (
        context_path.exists()
        and (directory / "tracts.geojson").exists()
        and (directory / "pois.csv.gz").exists()
    )
    if ready and not refresh:
        return json.loads(context_path.read_text())
    if city not in CITIES:
        raise KeyError(city)
    session = session or _session()
    bbox = city_bbox(city)
    if refresh:
        for path in [directory / "tracts.geojson", directory / "pois.csv.gz", context_path]:
            if path.exists():
                path.unlink()
        for category in POI_CATEGORIES:
            cache_path = directory / f"pois_{category}.csv.gz"
            if cache_path.exists():
                cache_path.unlink()
    print(f"{city}: census tracts", flush=True)
    if (directory / "tracts.geojson").exists():
        tract_info = _summarize_tracts(city)
    else:
        tract_info = build_tracts(session, city, bbox)
    print(f"{city}: OpenStreetMap", flush=True)
    poi_counts = build_pois(session, city, bbox)
    context = {
        "city": city,
        "bbox": [float(v) for v in bbox],
        "bbox_pad_degrees": BBOX_PAD_DEGREES,
        "acs_vintage": f"{ACS_YEAR} 5-year",
        "acs_source": (
            "U.S. Census Bureau ACS table-based summary file, tables "
            "B19013 (median household income), B25064 (median gross rent), "
            "B25077 (median home value), B01003 (total population)"
        ),
        "tiger_source": f"TIGER/Line {TIGER_YEAR} census tracts",
        "osm_source": "OpenStreetMap, queried through a public Overpass endpoint",
        "downloaded_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "census_medians": tract_info["census_medians"],
        "n_tracts": tract_info["n_tracts"],
        "acs_matched_income": tract_info.get("acs_matched_income"),
        "n_pois": poi_counts,
        "distance_cap_km": 20.0,
        "count_radius_km": 1.0,
        "nearest_tract_fallback_km": 2.0,
    }
    context_path.write_text(json.dumps(context, indent=2))
    return context


def ensure_external(cities: tuple[str, ...] | list[str], *, refresh: bool = False) -> None:
    session = _session()
    for city in cities:
        ensure_city(city, refresh=refresh, session=session)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download Census and OpenStreetMap extracts.")
    parser.add_argument("cities", nargs="*", help="City keys. Defaults to the study cities.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached extracts.")
    args = parser.parse_args()
    from data.cities import STUDY_CITIES

    cities = tuple(args.cities) if args.cities else STUDY_CITIES
    ensure_external(cities, refresh=args.refresh)
    print("External extracts are in data/external/", flush=True)


if __name__ == "__main__":
    main()
