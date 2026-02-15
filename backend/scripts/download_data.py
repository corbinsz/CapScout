"""
Download Inside Airbnb data for all registered cities.

Usage (from backend/ directory):
    python -m scripts.download_data            # download all cities
    python -m scripts.download_data nashville  # download one city
"""
import gzip
import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from cities import CITIES


IA_BASE = 'https://data.insideairbnb.com/united-states'
FILES = ['listings.csv.gz', 'calendar.csv.gz']

_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


def _download_file(url, dest):
    """Download a URL to a local path with progress."""
    print(f'  Downloading {url}')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': _UA})
        with urllib.request.urlopen(req) as resp, open(dest, 'wb') as f:
            shutil.copyfileobj(resp, f)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f'    -> {dest} ({size_mb:.1f} MB)')
        return True
    except Exception as e:
        print(f'    FAILED: {e}')
        return False


def _decompress_gz(gz_path, out_path):
    """Decompress a .gz file."""
    print(f'  Decompressing {os.path.basename(gz_path)}...')
    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'    -> {out_path} ({size_mb:.1f} MB)')


def download_city(city_key):
    """Download Inside Airbnb data for a single city."""
    city = CITIES.get(city_key)
    if not city:
        print(f'Unknown city: {city_key}')
        return False

    print(f"\n{'='*50}")
    print(f"Downloading: {city['display']}")
    print(f"{'='*50}")

    dest_dir = config.city_data_dir(city_key)
    state = city['ia_state']
    ia_city = city['ia_city']
    date = city['ia_date']

    success = True
    for fname in FILES:
        url = f'{IA_BASE}/{state}/{ia_city}/{date}/data/{fname}'
        gz_dest = os.path.join(dest_dir, fname)

        if not _download_file(url, gz_dest):
            success = False
            continue

        # Decompress listings.csv.gz -> listings.csv
        if fname == 'listings.csv.gz':
            csv_dest = os.path.join(dest_dir, 'listings.csv')
            _decompress_gz(gz_dest, csv_dest)

    return success


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    cities_to_download = args if args else list(CITIES.keys())

    for city_key in cities_to_download:
        download_city(city_key)

    print(f"\nDone! Downloaded data for {len(cities_to_download)} cities.")


if __name__ == '__main__':
    main()
