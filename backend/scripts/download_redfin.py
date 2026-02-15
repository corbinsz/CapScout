"""
Download all for-sale listings from Redfin for registered cities.

Uses the Redfin GIS JSON API which returns up to ~10k listings per request.
Parses the JSON response and saves as CSV matching the standard Redfin format.

Usage (from backend/ directory):
    python -m scripts.download_redfin            # all cities
    python -m scripts.download_redfin austin     # one city
"""
import csv
import json
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from cities import CITIES

_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
MAX_HOMES = 10000

# CSV columns to produce (matching Redfin CSV download format)
CSV_COLUMNS = [
    'SALE TYPE', 'SOLD DATE', 'PROPERTY TYPE', 'ADDRESS', 'CITY',
    'STATE OR PROVINCE', 'ZIP OR POSTAL CODE', 'PRICE', 'BEDS', 'BATHS',
    'LOCATION', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT', 'DAYS ON MARKET',
    '$/SQUARE FEET', 'HOA/MONTH', 'STATUS', 'NEXT OPEN HOUSE START TIME',
    'NEXT OPEN HOUSE END TIME', 'URL (SEE https://www.redfin.com/buy-a-home'
    '/comparative-market-analysis FOR INFO ON PRICING)', 'SOURCE', 'MLS#',
    'FAVORITE', 'INTERESTED', 'LATITUDE', 'LONGITUDE',
]

_PROP_TYPES = {
    1: 'Single Family Residential',
    2: 'Condo/Co-op',
    3: 'Townhouse',
    4: 'Multi-Family (2-4 Unit)',
    5: 'Multi-Family (5+ Unit)',
    6: 'Single Family Residential',
    7: 'Mobile/Manufactured Home',
    8: 'Vacant Land',
    13: 'Other',
}

_LISTING_TYPES = {
    1: 'MLS Listing',
    2: 'For-Sale-by-Owner Listing',
    3: 'New Construction Home',
    4: 'New Construction Plan',
}


def _parse_home(home):
    """Convert a GIS JSON home object to a flat CSV-style dict."""
    ll = home.get('latLong', {}).get('value', {})
    address = home.get('streetLine', {}).get('value', '')
    unit = home.get('unitNumber', {}).get('value', '')
    if unit:
        address = f'{address} {unit}'
    url_path = home.get('url', '')

    return {
        'SALE TYPE': _LISTING_TYPES.get(home.get('listingType', 0), 'MLS Listing'),
        'SOLD DATE': '',
        'PROPERTY TYPE': _PROP_TYPES.get(home.get('propertyType', 0), ''),
        'ADDRESS': address,
        'CITY': home.get('city', ''),
        'STATE OR PROVINCE': home.get('state', ''),
        'ZIP OR POSTAL CODE': home.get('zip', ''),
        'PRICE': home.get('price', {}).get('value', ''),
        'BEDS': home.get('beds', ''),
        'BATHS': home.get('baths', ''),
        'LOCATION': home.get('location', {}).get('value', ''),
        'SQUARE FEET': home.get('sqFt', {}).get('value', ''),
        'LOT SIZE': home.get('lotSize', {}).get('value', ''),
        'YEAR BUILT': home.get('yearBuilt', {}).get('value', ''),
        'DAYS ON MARKET': home.get('dom', {}).get('value', ''),
        '$/SQUARE FEET': home.get('pricePerSqFt', {}).get('value', ''),
        'HOA/MONTH': home.get('hoa', {}).get('value', '') or '',
        'STATUS': home.get('mlsStatus', 'Active'),
        'NEXT OPEN HOUSE START TIME': '',
        'NEXT OPEN HOUSE END TIME': '',
        'URL (SEE https://www.redfin.com/buy-a-home'
        '/comparative-market-analysis FOR INFO ON PRICING)':
            f'https://www.redfin.com{url_path}' if url_path else '',
        'SOURCE': '',
        'MLS#': home.get('mlsId', {}).get('value', ''),
        'FAVORITE': 'N',
        'INTERESTED': 'Y',
        'LATITUDE': ll.get('latitude', ''),
        'LONGITUDE': ll.get('longitude', ''),
    }


def download_city(city_key):
    """Download all Redfin for-sale listings for a city."""
    city = CITIES.get(city_key)
    if not city:
        print(f'Unknown city: {city_key}')
        return 0

    region_id = city.get('redfin_region_id')
    if not region_id:
        print(f'{city_key}: no redfin_region_id configured, skipping')
        return 0

    print(f"\n{'='*50}")
    print(f"Redfin download: {city['display']} (region {region_id})")
    print(f"{'='*50}")

    url = (
        f'https://www.redfin.com/stingray/api/gis?'
        f'al=1&num_homes={MAX_HOMES}'
        f'&ord=redfin-recommended-asc&page_number=1'
        f'&sf=1,2,3,5,6,7&status=9&uipt=1,2,3,4,5,6,7,8'
        f'&v=8&region_id={region_id}&region_type=6'
    )

    print(f'  Fetching up to {MAX_HOMES} listings...')
    req = urllib.request.Request(url, headers={'User-Agent': _UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode('utf-8')
    except Exception as e:
        print(f'  FAILED: {e}')
        return 0

    if text.startswith('{}&&'):
        text = text[4:]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f'  JSON parse error: {e}')
        return 0

    homes = data.get('payload', {}).get('homes', [])
    print(f'  API returned {len(homes)} listings')

    # Parse and deduplicate by address
    seen = set()
    unique = []
    for h in homes:
        row = _parse_home(h)
        addr = (row.get('ADDRESS') or '').strip()
        if addr and addr not in seen:
            seen.add(addr)
            unique.append(row)

    print(f'  After dedup: {len(unique)} unique listings')

    if not unique:
        print('  No listings found!')
        return 0

    # Write CSV
    dest = os.path.join(config.city_data_dir(city_key), 'redfin_listings.csv')
    with open(dest, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(unique)

    print(f'  Saved {len(unique)} listings to {dest}')
    return len(unique)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    cities_to_download = args if args else list(CITIES.keys())

    total = 0
    for city_key in cities_to_download:
        total += download_city(city_key)
        if len(cities_to_download) > 1:
            time.sleep(2)

    print(f'\nDone! {total} total listings across {len(cities_to_download)} cities.')


if __name__ == '__main__':
    main()
