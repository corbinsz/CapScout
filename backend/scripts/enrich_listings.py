"""
Enrich Inside Airbnb listings with county assessor property data.

Downloads assessor records (sqft, year_built, lot_size) and spatially
matches them to Airbnb listings using KD-tree nearest-neighbor search.

Usage (from backend/ directory):
    python -m scripts.enrich_listings            # all enrichable cities
    python -m scripts.enrich_listings seattle    # one city
    python -m scripts.enrich_listings --force    # re-download assessor data
"""
import json
import math
import os
import shutil
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from cities import CITIES

# ~200m at mid-latitudes (200 / 111000)
MATCH_RADIUS_DEG = 0.002
OUTPUT_FILENAME = 'assessor_enrichment.csv'
_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


# =====================================================================
# Download helpers
# =====================================================================

def _download(url, dest, force=False):
    """Download URL to file. Skip if already exists unless force=True."""
    if os.path.exists(dest) and not force:
        print(f'    Cached: {os.path.basename(dest)}')
        return True
    print(f'    Downloading {url[:100]}...')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': _UA})
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, 'wb') as f:
            shutil.copyfileobj(resp, f)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f'    -> {os.path.basename(dest)} ({size_mb:.1f} MB)')
        return True
    except Exception as e:
        print(f'    FAILED: {e}')
        return False


def _query_arcgis_parcels(service_url, bbox, out_sr=4326, batch_size=1000):
    """
    Query an ArcGIS MapServer/FeatureServer for parcels within a bounding box.

    Returns DataFrame with columns: pin, latitude, longitude
    Fetches parcel polygon envelopes and computes centroids.
    """
    query_url = f'{service_url}/query'
    all_features = []
    offset = 0

    # Use spatial query with bounding box envelope
    geom = json.dumps({
        'xmin': bbox[0], 'ymin': bbox[1],
        'xmax': bbox[2], 'ymax': bbox[3],
        'spatialReference': {'wkid': 4326},
    })

    while True:
        params = urllib.parse.urlencode({
            'where': '1=1',
            'geometry': geom,
            'geometryType': 'esriGeometryEnvelope',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': 'PIN,MAJOR,MINOR',
            'returnGeometry': 'true',
            'geometryPrecision': 6,
            'outSR': out_sr,
            'resultOffset': offset,
            'resultRecordCount': batch_size,
            'f': 'json',
        })

        url = f'{query_url}?{params}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': _UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f'    ArcGIS query failed at offset {offset}: {e}')
            break

        features = data.get('features', [])
        if not features:
            break

        for feat in features:
            attrs = feat.get('attributes', {})
            geom_data = feat.get('geometry', {})
            rings = geom_data.get('rings', [])

            if not rings:
                continue

            # Compute centroid from polygon rings (average of all vertices)
            all_pts = [pt for ring in rings for pt in ring]
            if not all_pts:
                continue

            cx = sum(p[0] for p in all_pts) / len(all_pts)
            cy = sum(p[1] for p in all_pts) / len(all_pts)

            pin = attrs.get('PIN', '')
            if not pin:
                major = str(attrs.get('MAJOR', '')).strip()
                minor = str(attrs.get('MINOR', '')).strip()
                pin = f'{major}{minor}'

            all_features.append({
                'pin': pin,
                'latitude': round(cy, 6),
                'longitude': round(cx, 6),
            })

        print(f'    Fetched {len(all_features)} parcels...', end='\r')
        offset += len(features)

        if not data.get('exceededTransferLimit', False) and len(features) < batch_size:
            break

        time.sleep(0.2)  # polite rate limiting

    print(f'    Fetched {len(all_features)} parcels total')
    return pd.DataFrame(all_features)


# =====================================================================
# County-specific parsers
# =====================================================================

def _check_manual_files(assessor_dir, filenames, download_url):
    """Check if manually-downloaded assessor files exist. Print instructions if not."""
    missing = []
    for f in filenames:
        if not os.path.exists(os.path.join(assessor_dir, f)):
            missing.append(f)
    if missing:
        print(f'  Missing assessor files in {assessor_dir}:')
        for f in missing:
            print(f'    - {f}')
        print(f'  Download from: {download_url}')
        print(f'  Place files in: {assessor_dir}')
        return False
    return True


def _load_king_county(city_key, assessor_dir, force=False):
    """
    King County (Seattle): Load EXTR_ResBldg.csv + EXTR_Parcel.csv,
    query ArcGIS for parcel centroids, join everything.

    Returns DataFrame with: latitude, longitude, sqft, year_built, bedrooms, lot_sqft
    """
    city = CITIES[city_key]
    assessor_cfg = city['assessor']

    # Step 1: Check for manually-downloaded assessor CSVs
    if not _check_manual_files(assessor_dir, assessor_cfg['files'], assessor_cfg['download_url']):
        return None

    # Step 2: Load building data
    bldg_path = os.path.join(assessor_dir, 'EXTR_ResBldg.csv')
    print('  Loading residential building data...')
    bldg = pd.read_csv(bldg_path, low_memory=False, encoding='latin-1')

    # Normalize column names (King County uses mixed case)
    bldg.columns = [c.strip() for c in bldg.columns]

    # Build parcel key from Major+Minor
    bldg['pin'] = (bldg['Major'].astype(str).str.strip().str.zfill(6) +
                   bldg['Minor'].astype(str).str.strip().str.zfill(4))

    # Keep one building per parcel (largest sqft if multiple)
    bldg = bldg.sort_values('SqFtTotLiving', ascending=False).drop_duplicates('pin', keep='first')
    print(f'    {len(bldg)} unique residential parcels')

    # Step 3: Load parcel data (lot size)
    parcel_path = os.path.join(assessor_dir, 'EXTR_Parcel.csv')
    print('  Loading parcel data...')
    parcel = pd.read_csv(parcel_path, low_memory=False, encoding='latin-1')
    parcel.columns = [c.strip() for c in parcel.columns]
    parcel['pin'] = (parcel['Major'].astype(str).str.strip().str.zfill(6) +
                     parcel['Minor'].astype(str).str.strip().str.zfill(4))

    # Step 4: Query ArcGIS for parcel centroids within the city bbox
    centroids_cache = os.path.join(assessor_dir, 'parcel_centroids.csv')
    if os.path.exists(centroids_cache) and not force:
        print('  Loading cached parcel centroids...')
        centroids = pd.read_csv(centroids_cache)
    else:
        print('  Querying ArcGIS for parcel centroids...')
        # Build bounding box from Airbnb listings + buffer
        listings_path = os.path.join(config.city_data_dir(city_key), 'listings.csv')
        listings = pd.read_csv(listings_path, usecols=['latitude', 'longitude'], low_memory=False)
        lat_min = listings['latitude'].min() - 0.02
        lat_max = listings['latitude'].max() + 0.02
        lng_min = listings['longitude'].min() - 0.02
        lng_max = listings['longitude'].max() + 0.02
        bbox = [lng_min, lat_min, lng_max, lat_max]

        centroids = _query_arcgis_parcels(assessor_cfg['arcgis_parcels'], bbox)
        if len(centroids) > 0:
            centroids.to_csv(centroids_cache, index=False)
            print(f'    Cached centroids to {os.path.basename(centroids_cache)}')

    if len(centroids) == 0:
        print('  No parcel centroids found!')
        return None

    # Step 5: Join building + parcel + centroids on PIN
    # Column names vary in case across King County data vintages
    bldg.columns = [c.strip() for c in bldg.columns]
    col_map = {c.lower(): c for c in bldg.columns}
    bath3q = col_map.get('bath3qtrcount', 'Bath3QtrCount')
    merged = bldg[['pin', 'SqFtTotLiving', 'YrBuilt', 'Bedrooms',
                    'BathFullCount', bath3q, 'BathHalfCount']].copy()
    merged = merged.rename(columns={bath3q: 'Bath3QtrCount'})
    merged = merged.merge(parcel[['pin', 'SqFtLot']], on='pin', how='left')
    merged = merged.merge(centroids, on='pin', how='inner')

    # Normalize column names
    merged = merged.rename(columns={
        'SqFtTotLiving': 'sqft',
        'YrBuilt': 'year_built',
        'Bedrooms': 'bedrooms',
        'SqFtLot': 'lot_sqft',
    })

    # Compute total bathrooms
    merged['bathrooms'] = (
        merged.get('BathFullCount', 0).fillna(0) +
        merged.get('Bath3QtrCount', 0).fillna(0) * 0.75 +
        merged.get('BathHalfCount', 0).fillna(0) * 0.5
    )

    print(f'    {len(merged)} parcels with coordinates + building data')
    return merged[['latitude', 'longitude', 'sqft', 'year_built',
                    'bedrooms', 'bathrooms', 'lot_sqft']].copy()


def _query_arcgis_centroids(service_url, bbox, id_field='PROP_ID',
                            batch_size=1000):
    """
    Query an ArcGIS service for parcel centroids from polygon geometry.

    Requests polygons in WGS84 (inSR/outSR=4326) and computes centroids
    from ring vertices. Works with services in any coordinate system.

    Returns DataFrame with columns: prop_id, latitude, longitude
    """
    import urllib.parse as urlparse
    query_url = f'{service_url}/query'
    all_rows = []
    offset = 0

    geom = json.dumps({
        'xmin': bbox[0], 'ymin': bbox[1],
        'xmax': bbox[2], 'ymax': bbox[3],
        'spatialReference': {'wkid': 4326},
    })

    while True:
        params = urlparse.urlencode({
            'where': '1=1',
            'geometry': geom,
            'geometryType': 'esriGeometryEnvelope',
            'spatialRel': 'esriSpatialRelIntersects',
            'inSR': 4326,
            'outSR': 4326,
            'outFields': id_field,
            'returnGeometry': 'true',
            'geometryPrecision': 6,
            'resultOffset': offset,
            'resultRecordCount': batch_size,
            'f': 'json',
        })
        url = f'{query_url}?{params}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': _UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f'    ArcGIS query failed at offset {offset}: {e}')
            break

        features = data.get('features', [])
        if not features:
            break

        for feat in features:
            attrs = feat.get('attributes', {})
            pid = attrs.get(id_field)
            if not pid:
                continue

            geom_data = feat.get('geometry', {})
            rings = geom_data.get('rings', [])
            if not rings:
                continue

            all_pts = [pt for ring in rings for pt in ring]
            if not all_pts:
                continue

            cx = sum(p[0] for p in all_pts) / len(all_pts)
            cy = sum(p[1] for p in all_pts) / len(all_pts)

            all_rows.append({
                'prop_id': int(pid),
                'latitude': round(cy, 6),
                'longitude': round(cx, 6),
            })

        print(f'    Fetched {len(all_rows)} parcels...', end='\r')
        offset += len(features)

        if not data.get('exceededTransferLimit', False) and len(features) < batch_size:
            break
        time.sleep(0.2)

    print(f'    Fetched {len(all_rows)} parcels total')
    return pd.DataFrame(all_rows)


def _load_travis_cad(city_key, assessor_dir, force=False):
    """
    Travis CAD (Austin): Load improvement_detail CSVs for sqft, year_built,
    beds, baths. Query ArcGIS for parcel centroids.

    Returns DataFrame with: latitude, longitude, sqft, year_built, bedrooms, bathrooms, lot_sqft
    """
    city = CITIES[city_key]
    assessor_cfg = city['assessor']

    # Step 1: Check for improvement detail CSVs
    detail_files = [f for f in os.listdir(assessor_dir)
                    if f.startswith('improvement_detail_') and f.endswith('.csv')]
    if not detail_files:
        if not _check_manual_files(assessor_dir, assessor_cfg.get('files', []),
                                   assessor_cfg['download_url']):
            return None

    # Step 2: Load all improvement detail CSVs
    print('  Loading improvement detail data...')
    dfs = []
    for f in sorted(detail_files):
        dfs.append(pd.read_csv(os.path.join(assessor_dir, f), low_memory=False))
    detail = pd.concat(dfs, ignore_index=True)

    # Filter to residential properties (state code A*)
    detail = detail[detail['stateCd'].str.startswith('A', na=False)].copy()
    print(f'    {len(detail)} residential improvement detail rows')

    # Step 3: Aggregate by property ID
    # Living area: sum of floor types
    living_types = {'1ST', '2ND', '3RD', '1/2', 'RSBLW'}
    living = detail[detail['imprvDetailType'].isin(living_types)]
    sqft_by_prop = living.groupby('pID')['area'].sum().rename('sqft')
    year_by_prop = living.groupby('pID')['actualYearBuilt'].max().rename('year_built')

    # Bedrooms (type 252), Bathrooms (type 251), Half baths (type 250)
    beds = detail[detail['imprvDetailType'] == '252'].groupby('pID')['area'].sum().rename('bedrooms')
    full_baths = detail[detail['imprvDetailType'] == '251'].groupby('pID')['area'].sum()
    half_baths = detail[detail['imprvDetailType'] == '250'].groupby('pID')['area'].sum()
    baths = (full_baths.add(half_baths * 0.5, fill_value=0)).rename('bathrooms')

    # Lot size from LAND_DET.TXT if available
    lot_sqft = pd.Series(dtype=float, name='lot_sqft')
    land_path = os.path.join(assessor_dir, 'LAND_DET.TXT')
    if os.path.exists(land_path):
        try:
            # LAND_DET.TXT is fixed-width: prop_id(12), year(4), ..., size fields
            # Try loading as CSV first in case it has delimiters
            land = pd.read_fwf(land_path, colspecs=[(0, 12), (12, 16)],
                               names=['pID', 'year'], encoding='latin-1', nrows=5)
            # Fall back to just using tcad_acres from ArcGIS
        except Exception:
            pass

    # Combine all property-level stats
    props = pd.DataFrame(index=sqft_by_prop.index)
    props = props.join(sqft_by_prop).join(year_by_prop).join(beds).join(baths)
    props = props.reset_index()
    props = props.rename(columns={'pID': 'prop_id'})
    props['lot_sqft'] = np.nan  # will fill from ArcGIS if available
    print(f'    {len(props)} unique residential properties')

    # Step 4: Get coordinates from ArcGIS
    centroids_cache = os.path.join(assessor_dir, 'parcel_centroids.csv')
    if os.path.exists(centroids_cache) and not force:
        print('  Loading cached parcel centroids...')
        centroids = pd.read_csv(centroids_cache)
    else:
        print('  Querying Travis County ArcGIS for parcel centroids...')
        listings_path = os.path.join(config.city_data_dir(city_key), 'listings.csv')
        listings = pd.read_csv(listings_path, usecols=['latitude', 'longitude'], low_memory=False)
        bbox = [
            listings['longitude'].min() - 0.02,
            listings['latitude'].min() - 0.02,
            listings['longitude'].max() + 0.02,
            listings['latitude'].max() + 0.02,
        ]

        arcgis_url = 'https://gis.traviscountytx.gov/server1/rest/services/Boundaries_and_Jurisdictions/TCAD_public/MapServer/0'
        centroids = _query_arcgis_centroids(arcgis_url, bbox)
        if len(centroids) > 0:
            # Also grab lot_sqft from tcad_acres
            centroids.to_csv(centroids_cache, index=False)
            print(f'    Cached centroids to {os.path.basename(centroids_cache)}')

    if len(centroids) == 0:
        print('  No parcel centroids found!')
        return None

    # Step 5: Join property data with centroids
    merged = props.merge(centroids, on='prop_id', how='inner')
    print(f'    {len(merged)} properties with coordinates')

    return merged[['latitude', 'longitude', 'sqft', 'year_built',
                    'bedrooms', 'bathrooms', 'lot_sqft']].copy()


def _load_dallas_cad(city_key, assessor_dir, force=False):
    """
    Dallas CAD: Load DCAD CSV data for sqft, year_built, beds, baths.
    Query ArcGIS for parcel centroids.

    Returns DataFrame with: latitude, longitude, sqft, year_built, bedrooms, bathrooms, lot_sqft
    """
    city = CITIES[city_key]
    assessor_cfg = city['assessor']

    # Look for extracted CSV files from DCAD
    csv_files = [f for f in os.listdir(assessor_dir) if f.endswith('.csv')]
    if not csv_files:
        zip_files = [f for f in os.listdir(assessor_dir) if f.endswith('.zip') or f.endswith('.ZIP')]
        if zip_files:
            import zipfile
            for zf in zip_files:
                zpath = os.path.join(assessor_dir, zf)
                try:
                    with zipfile.ZipFile(zpath) as z:
                        z.extractall(assessor_dir)
                        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                except Exception:
                    pass
        if not csv_files:
            if not _check_manual_files(assessor_dir, assessor_cfg.get('files', []),
                                       assessor_cfg['download_url']):
                return None

    # DCAD has multiple CSV files; find the residential building one
    print(f'  Available files: {csv_files[:10]}')

    # Try to find and load building detail data
    # DCAD typically has: res_detail.csv, building_res.csv, or similar
    bldg_df = None
    for f in csv_files:
        fpath = os.path.join(assessor_dir, f)
        try:
            sample = pd.read_csv(fpath, nrows=3, low_memory=False, encoding='latin-1')
            cols_lower = [c.lower() for c in sample.columns]
            # Look for file with sqft/year_built/bedrooms columns
            has_sqft = any('area' in c or 'sqft' in c or 'sq_ft' in c for c in cols_lower)
            has_yr = any('year' in c or 'yr_' in c for c in cols_lower)
            has_bed = any('bed' in c for c in cols_lower)
            if has_sqft and has_yr and has_bed:
                print(f'  Loading {f} (has sqft + year + bedrooms)...')
                bldg_df = pd.read_csv(fpath, low_memory=False, encoding='latin-1')
                print(f'    Columns: {list(bldg_df.columns)[:15]}...')
                break
        except Exception:
            continue

    if bldg_df is None:
        print('  Could not find residential building data CSV.')
        print('  Looking for any CSV with property characteristics...')
        # Try loading the largest CSV as the main data file
        csv_files_with_size = [(f, os.path.getsize(os.path.join(assessor_dir, f)))
                               for f in csv_files]
        csv_files_with_size.sort(key=lambda x: x[1], reverse=True)
        for f, sz in csv_files_with_size[:3]:
            print(f'    {f}: {sz/1024/1024:.1f} MB')
            try:
                sample = pd.read_csv(os.path.join(assessor_dir, f), nrows=5,
                                     low_memory=False, encoding='latin-1')
                print(f'    Columns: {list(sample.columns)[:20]}')
            except Exception as e:
                print(f'    Error: {e}')
        return None

    return bldg_df


# Source dispatcher — add new loaders here as counties are implemented
_LOADERS = {
    'king_county': _load_king_county,
    'travis_cad': _load_travis_cad,
    'dallas_cad': _load_dallas_cad,
}


# =====================================================================
# Spatial matching engine
# =====================================================================

def spatial_match(listings_df, assessor_df, radius_deg=MATCH_RADIUS_DEG):
    """
    Match Airbnb listings to assessor parcels using KD-tree.

    For each Airbnb listing, find the nearest assessor parcel within the radius.
    Prefer matches with the same bedroom count to disambiguate.

    Returns DataFrame with: listing_id, sqft, year_built, lot_sqft,
                            match_distance_m, bedroom_matched
    """
    assessor_valid = assessor_df.dropna(subset=['latitude', 'longitude']).copy()
    listings_valid = listings_df.dropna(subset=['latitude', 'longitude']).copy()

    if len(assessor_valid) == 0 or len(listings_valid) == 0:
        return pd.DataFrame()

    # Build KD-tree from assessor coordinates
    # At mid-latitudes, using raw lat/lng is fine for ~200m matching
    assessor_coords = assessor_valid[['latitude', 'longitude']].values
    tree = cKDTree(assessor_coords)

    listing_coords = listings_valid[['latitude', 'longitude']].values

    # Query all neighbors within radius for all listings at once
    matches = tree.query_ball_point(listing_coords, r=radius_deg)

    results = []
    for i, candidate_indices in enumerate(matches):
        if not candidate_indices:
            continue

        listing_id = listings_valid.iloc[i]['id']
        listing_beds = listings_valid.iloc[i].get('bedrooms', 0)
        if pd.isna(listing_beds):
            listing_beds = 0

        candidates = assessor_valid.iloc[candidate_indices]

        # Compute distances
        dists = np.sqrt(
            (candidates['latitude'].values - listing_coords[i, 0]) ** 2 +
            (candidates['longitude'].values - listing_coords[i, 1]) ** 2
        )

        # Prefer bedroom-matching candidates
        bed_mask = (candidates['bedrooms'].values == listing_beds)

        if bed_mask.any():
            bed_dists = np.where(bed_mask, dists, np.inf)
            best_local = np.argmin(bed_dists)
            best_idx = candidate_indices[best_local]
            best_dist = dists[best_local]
            bed_matched = True
        else:
            best_local = np.argmin(dists)
            best_idx = candidate_indices[best_local]
            best_dist = dists[best_local]
            bed_matched = False

        row = assessor_valid.iloc[best_idx]
        dist_m = best_dist * 111_000  # approximate degrees to meters

        results.append({
            'listing_id': listing_id,
            'sqft': row['sqft'] if pd.notna(row['sqft']) and row['sqft'] > 0 else np.nan,
            'year_built': row['year_built'] if pd.notna(row['year_built']) and row['year_built'] > 1800 else np.nan,
            'lot_sqft': row['lot_sqft'] if pd.notna(row.get('lot_sqft')) and row.get('lot_sqft', 0) > 0 else np.nan,
            'match_distance_m': round(dist_m, 1),
            'bedroom_matched': bed_matched,
        })

    return pd.DataFrame(results)


# =====================================================================
# Main enrichment pipeline
# =====================================================================

def enrich_city(city_key, force=False):
    """Run full enrichment pipeline for one city."""
    city = CITIES.get(city_key)
    if not city:
        print(f'Unknown city: {city_key}')
        return False

    assessor_cfg = city.get('assessor')
    if not assessor_cfg:
        print(f'{city_key}: no assessor data configured, skipping')
        return False

    print(f"\n{'='*50}")
    print(f"Enriching: {city['display']}")
    print(f"{'='*50}")

    data_dir = config.city_data_dir(city_key)
    assessor_dir = os.path.join(data_dir, 'assessor_raw')
    os.makedirs(assessor_dir, exist_ok=True)

    # Step 1: Load assessor data (county-specific)
    source = assessor_cfg['source']
    loader = _LOADERS.get(source)
    if not loader:
        print(f'  No loader implemented for: {source}')
        print(f'  Available: {list(_LOADERS.keys())}')
        return False

    print(f'  Source: {source}')
    assessor_df = loader(city_key, assessor_dir, force=force)
    if assessor_df is None or len(assessor_df) == 0:
        print(f'  Could not load assessor data.')
        return False

    print(f'  {len(assessor_df)} assessor records loaded')

    # Step 2: Load Airbnb listings
    listings_path = os.path.join(data_dir, 'listings.csv')
    if not os.path.exists(listings_path):
        print(f'  No listings.csv found. Run: python -m scripts.download_data {city_key}')
        return False

    listings = pd.read_csv(listings_path, low_memory=False,
                           usecols=['id', 'latitude', 'longitude', 'bedrooms', 'room_type'])
    listings = listings[listings['room_type'] == 'Entire home/apt'].copy()
    listings['bedrooms'] = pd.to_numeric(listings['bedrooms'], errors='coerce').fillna(1)
    print(f'  {len(listings)} Airbnb listings (entire home/apt)')

    # Step 3: Spatial match
    print(f'  Spatial matching (radius ~{MATCH_RADIUS_DEG * 111_000:.0f}m)...')
    enrichment = spatial_match(listings, assessor_df)

    if len(enrichment) == 0:
        print('  No matches found!')
        return False

    # Step 4: Report stats
    n_matched = len(enrichment)
    n_sqft = enrichment['sqft'].notna().sum()
    n_year = enrichment['year_built'].notna().sum()
    n_lot = enrichment['lot_sqft'].notna().sum()
    n_bed_match = enrichment['bedroom_matched'].sum()
    avg_dist = enrichment['match_distance_m'].mean()

    print(f'  Matched {n_matched}/{len(listings)} ({n_matched/len(listings)*100:.1f}%)')
    print(f'    sqft: {n_sqft} | year_built: {n_year} | lot_sqft: {n_lot}')
    print(f'    Bedroom-matched: {n_bed_match} ({n_bed_match/n_matched*100:.0f}%)')
    print(f'    Avg match distance: {avg_dist:.0f}m')

    # Step 5: Save enrichment CSV
    output_path = os.path.join(data_dir, OUTPUT_FILENAME)
    enrichment.to_csv(output_path, index=False)
    print(f'  Saved: {output_path}')

    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    force = '--force' in sys.argv

    if args:
        cities = args
    else:
        cities = [k for k, v in CITIES.items() if v.get('assessor')]

    if not cities:
        print('No cities with assessor data configured.')
        return

    print(f'Cities to enrich: {", ".join(cities)}')
    for city_key in cities:
        enrich_city(city_key, force=force)

    print(f'\nDone! Processed {len(cities)} cities.')


if __name__ == '__main__':
    import urllib.parse
    main()
