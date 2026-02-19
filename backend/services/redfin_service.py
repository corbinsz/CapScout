"""
Redfin CSV import service.
Parses CSV files downloaded from Redfin's "Download All" button and
normalizes them to the unified property shape used by the for-sale pipeline.
"""
import csv
import hashlib
import math
import os
from urllib.parse import quote_plus

import config
from cities import DEFAULT_CITY

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
CSV_PATH = os.path.join(DATA_DIR, 'redfin_listings.csv')


def _csv_path(city_key=None):
    """Return the Redfin CSV path for a city."""
    if city_key:
        return os.path.join(config.city_data_dir(city_key), 'redfin_listings.csv')
    return CSV_PATH

# Columns we expect in a Redfin CSV download
REQUIRED_COLUMNS = {'ADDRESS', 'CITY', 'STATE OR PROVINCE', 'ZIP OR POSTAL CODE', 'PRICE', 'BEDS', 'BATHS'}


def has_data(city_key=None):
    """Return True if a saved Redfin CSV exists on disk."""
    return os.path.isfile(_csv_path(city_key))


def save_csv(file_storage, city_key=None):
    """Save an uploaded Redfin CSV (werkzeug FileStorage) to disk.

    Returns the number of data rows in the file.
    Raises ValueError if the file doesn't look like a Redfin CSV.
    """
    path = _csv_path(city_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = file_storage.read().decode('utf-8-sig')  # Redfin uses UTF-8 BOM
    reader = csv.DictReader(text.splitlines())

    if not reader.fieldnames:
        raise ValueError('Uploaded file has no columns.')

    headers = {h.strip().upper() for h in reader.fieldnames if h}
    missing = REQUIRED_COLUMNS - headers
    if missing:
        raise ValueError(f'Missing expected Redfin columns: {", ".join(sorted(missing))}')

    # Write sanitized text to disk
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)

    # Invalidate cache for this city
    key = city_key or '__default__'
    _listings_cache.pop(key, None)

    # Count rows
    rows = list(csv.DictReader(text.splitlines()))
    return len(rows)


_listings_cache = {}


def get_listings(city_key=None):
    """Read the saved CSV and return a list of normalized dicts (cached)."""
    key = city_key or '__default__'
    if key in _listings_cache:
        return _listings_cache[key]
    if not has_data(city_key):
        return []
    with open(_csv_path(city_key), 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        listings = [normalize_listing(_upper_keys(row)) for row in reader
                    if _valid_row(_upper_keys(row))]
    _listings_cache[key] = listings
    return listings


def _upper_keys(row):
    """Normalize all CSV column names to uppercase for case-insensitive matching."""
    return {k.strip().upper(): v for k, v in row.items()}


# Property types not suitable for STR investment
_EXCLUDED_TYPES = {
    'vacant land', 'other', 'parking', 'commercial',
    'multi-family (5+ unit)',
}

# Listing types that are plans/blueprints, not actual properties
_EXCLUDED_SALE_TYPES = {
    'new construction plan',
}


def _valid_row(row):
    """Skip rows that aren't viable STR investment properties."""
    addr = row.get('ADDRESS') or ''
    price = row.get('PRICE') or ''
    if not addr.strip() or not price.strip():
        return False

    # Filter out non-residential property types
    prop_type = (row.get('PROPERTY TYPE') or '').strip().lower()
    if prop_type in _EXCLUDED_TYPES:
        return False

    # Filter out construction plans (not actual properties)
    sale_type = (row.get('SALE TYPE') or '').strip().lower()
    if sale_type in _EXCLUDED_SALE_TYPES:
        return False

    # Must have at least 1 bedroom for STR
    beds = 0
    try:
        beds = int(float(str(row.get('BEDS', 0)).replace(',', '').strip() or 0))
    except (ValueError, TypeError):
        pass
    if beds < 1:
        return False

    return True


def _safe_float(val, default=0):
    if val is None:
        return default
    try:
        return float(str(val).replace(',', '').strip())
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    if val is None:
        return default
    try:
        return int(float(str(val).replace(',', '').strip()))
    except (ValueError, TypeError):
        return default


def _lat_lng_to_tile(lat, lng, zoom=17):
    """Convert lat/lng to OSM tile coordinates."""
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return zoom, x, y


def _property_image_url(lat, lng, address):
    """Build a property image URL (OSM fallback only).

    Street View photos are added later by the route layer, only for
    properties that score above the investment threshold.
    """
    if lat and lng:
        z, x, y = _lat_lng_to_tile(lat, lng, zoom=17)
        return f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"

    return None


def normalize_listing(row):
    """Map a Redfin CSV row to the unified property dict."""
    address = (row.get('ADDRESS') or '').strip()
    city = (row.get('CITY') or '').strip()
    state = (row.get('STATE OR PROVINCE') or '').strip()
    zipcode = (row.get('ZIP OR POSTAL CODE') or '').strip()

    full_address = ', '.join(part for part in [address, city, state, zipcode] if part)

    # Deterministic id from address
    row_id = hashlib.md5(full_address.encode()).hexdigest()[:12]

    lat = _safe_float(row.get('LATITUDE'))
    lng = _safe_float(row.get('LONGITUDE'))

    return {
        'id': f'redfin-{row_id}',
        'name': full_address,
        'address': full_address,
        'photo_url': _property_image_url(lat, lng, full_address),
        'price': _safe_float(row.get('PRICE')),
        'bedrooms': _safe_int(row.get('BEDS')),
        'bathrooms': _safe_int(row.get('BATHS')),
        'sqft': _safe_int(row.get('SQUARE FEET')),
        'year_built': _safe_int(row.get('YEAR BUILT')),
        'lot_size': _safe_int(row.get('LOT SIZE')) if _safe_int(row.get('LOT SIZE')) <= 217800 else 0,  # cap 5 acres; Redfin shows complex lot for condos
        'days_on_market': _safe_int(row.get('DAYS ON MARKET')),
        'hoa_monthly': _safe_float(row.get('HOA/MONTH')),
        'price_per_sqft': _safe_float(row.get('$/SQUARE FEET')),
        'latitude': lat,
        'longitude': lng,
        'property_type': row.get('PROPERTY TYPE', ''),
        'url': row.get('URL (SEE HTTPS://WWW.REDFIN.COM/BUY-A-HOME/COMPARATIVE-MARKET-ANALYSIS FOR INFO ON PRICING)',
                        row.get('URL', '')),
        'source': 'redfin',
    }


def row_count(city_key=None):
    """Return the number of listings in the saved CSV, or 0."""
    if not has_data(city_key):
        return 0
    with open(_csv_path(city_key), 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return sum(1 for row in reader if _valid_row(_upper_keys(row)))
