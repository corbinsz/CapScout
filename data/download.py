"""Download Inside Airbnb listing snapshots.

Inside Airbnb publishes these files under CC BY 4.0. The gzip files are
gitignored. A fresh checkout does not contain them. Refreshing a city
rewrites data/raw/cities/<city>/listings.csv.gz. Calendar and review dumps
are intentionally not downloaded: the calendar does not separate a booked
night from a host-blocked night, and listing-level review totals are already
on the listings file.

The scored study uses Austin 2025-09-16, San Diego 2025-03-16, and Seattle
2025-03-16. Those dates live in data/cities.py. The Austin file dated
2025-06-13 on Inside Airbnb is a different scrape and will not match the
frozen ids.

Usage (from the repo root):
    python -m data.download austin san-diego seattle
    python -m data.download
"""

from __future__ import annotations

import sys
import urllib.request

from data.cities import CITIES
from data.ingest import raw_city_dir

IA_BASE = "https://data.insideairbnb.com/united-states"
_UA = "CapScout personal research (CC BY 4.0 Inside Airbnb snapshot)"


def download_city(city_key: str) -> bool:
    """Download one city's listings.csv.gz. Returns False on failure."""
    city = CITIES.get(city_key)
    if not city:
        print(f"Unknown city: {city_key}", file=sys.stderr)
        return False

    dest_dir = raw_city_dir(city_key)
    url = (
        f"{IA_BASE}/{city['ia_state']}/{city['ia_city']}/{city['ia_date']}" "/data/listings.csv.gz"
    )
    dest = dest_dir / "listings.csv.gz"
    print(f"Downloading {city['display']}: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
            out.write(resp.read())
    except Exception as exc:  # network and HTTP errors should not crash a batch
        print(f"  FAILED: {exc}", file=sys.stderr)
        return False
    print(f"  wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def main(argv: list[str] | None = None) -> None:
    args = [a for a in (argv if argv is not None else sys.argv[1:]) if not a.startswith("-")]
    keys = args or list(CITIES.keys())
    failed = [key for key in keys if not download_city(key)]
    if failed:
        raise SystemExit(f"Download failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
