"""
Redfin CSV import service.
Parses CSV files downloaded from Redfin's "Download All" button and
normalizes them to the unified property shape used by the for-sale pipeline.
"""
import csv
import hashlib
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
CSV_PATH = os.path.join(DATA_DIR, 'redfin_listings.csv')

# Columns we expect in a Redfin CSV download
REQUIRED_COLUMNS = {'ADDRESS', 'CITY', 'STATE OR PROVINCE', 'ZIP OR POSTAL CODE', 'PRICE', 'BEDS', 'BATHS'}


def has_data():
    """Return True if a saved Redfin CSV exists on disk."""
    return os.path.isfile(CSV_PATH)


def save_csv(file_storage):
    """Save an uploaded Redfin CSV (werkzeug FileStorage) to disk.

    Returns the number of data rows in the file.
    Raises ValueError if the file doesn't look like a Redfin CSV.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    text = file_storage.read().decode('utf-8-sig')  # Redfin uses UTF-8 BOM
    reader = csv.DictReader(text.splitlines())

    if not reader.fieldnames:
        raise ValueError('Uploaded file has no columns.')

    headers = {h.strip().upper() for h in reader.fieldnames if h}
    missing = REQUIRED_COLUMNS - headers
    if missing:
        raise ValueError(f'Missing expected Redfin columns: {", ".join(sorted(missing))}')

    # Write sanitized text to disk
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        f.write(text)

    # Count rows
    rows = list(csv.DictReader(text.splitlines()))
    return len(rows)


def get_listings():
    """Read the saved CSV and return a list of normalized dicts."""
    if not has_data():
        return []
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return [normalize_listing(_upper_keys(row)) for row in reader
                if _valid_row(_upper_keys(row))]


def _upper_keys(row):
    """Normalize all CSV column names to uppercase for case-insensitive matching."""
    return {k.strip().upper(): v for k, v in row.items()}


def _valid_row(row):
    """Skip rows with no address or no price."""
    addr = row.get('ADDRESS') or ''
    price = row.get('PRICE') or ''
    return bool(addr.strip()) and bool(price.strip())


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


def normalize_listing(row):
    """Map a Redfin CSV row to the unified property dict."""
    address = (row.get('ADDRESS') or '').strip()
    city = (row.get('CITY') or '').strip()
    state = (row.get('STATE OR PROVINCE') or '').strip()
    zipcode = (row.get('ZIP OR POSTAL CODE') or '').strip()

    full_address = ', '.join(part for part in [address, city, state, zipcode] if part)

    # Deterministic id from address
    row_id = hashlib.md5(full_address.encode()).hexdigest()[:12]

    return {
        'id': f'redfin-{row_id}',
        'name': full_address,
        'address': full_address,
        'photo_url': None,
        'price': _safe_float(row.get('PRICE')),
        'bedrooms': _safe_int(row.get('BEDS')),
        'bathrooms': _safe_int(row.get('BATHS')),
        'sqft': _safe_int(row.get('SQUARE FEET')),
        'latitude': _safe_float(row.get('LATITUDE')),
        'longitude': _safe_float(row.get('LONGITUDE')),
        'property_type': row.get('PROPERTY TYPE', ''),
        'url': row.get('URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)',
                        row.get('URL', '')),
        'source': 'redfin',
    }


def row_count():
    """Return the number of listings in the saved CSV, or 0."""
    if not has_data():
        return 0
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return sum(1 for row in reader if _valid_row(_upper_keys(row)))
