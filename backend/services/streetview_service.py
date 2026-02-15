"""
Google Street View image fetching with disk caching and monthly rate limiting.

Images are cached permanently to disk. A monthly counter prevents exceeding
the Google Maps free tier (~28,500 requests/month at $7/1000).
"""
import hashlib
import json
import os
from datetime import datetime

import requests

import config

MONTHLY_LIMIT = 25000
STREETVIEW_API_URL = 'https://maps.googleapis.com/maps/api/streetview'


def _cache_dir():
    return config.streetview_cache_dir()


def _counter_path():
    return os.path.join(_cache_dir(), 'counter.json')


def _image_path(image_hash):
    return os.path.join(_cache_dir(), f'{image_hash}.jpg')


def _noimage_path(image_hash):
    return os.path.join(_cache_dir(), f'{image_hash}.noimage')


def _make_hash(lat, lng):
    """Deterministic hash from lat/lng for cache key."""
    key = f'{lat:.6f},{lng:.6f}'
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_counter():
    """Load the monthly request counter from disk."""
    path = _counter_path()
    if not os.path.isfile(path):
        return {'month': '', 'count': 0}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {'month': '', 'count': 0}


def _save_counter(data):
    with open(_counter_path(), 'w') as f:
        json.dump(data, f)


def _current_month():
    return datetime.now().strftime('%Y-%m')


def _is_under_limit():
    counter = _load_counter()
    month = _current_month()
    if counter.get('month') != month:
        return True
    return counter.get('count', 0) < MONTHLY_LIMIT


def _increment_counter():
    counter = _load_counter()
    month = _current_month()
    if counter.get('month') != month:
        counter = {'month': month, 'count': 0}
    counter['count'] = counter.get('count', 0) + 1
    _save_counter(counter)


def get_streetview_image(lat, lng):
    """Fetch a Street View image, returning the file path or None.

    Uses disk cache — only hits the Google API on first request for a location.
    Returns None if no API key, over monthly limit, or no Street View coverage.
    """
    if not lat or not lng:
        return None

    image_hash = _make_hash(lat, lng)

    # Check cache first
    cached = _image_path(image_hash)
    if os.path.isfile(cached):
        return cached

    # Check no-image marker
    if os.path.isfile(_noimage_path(image_hash)):
        return None

    # Check API key and rate limit
    api_key = config.GOOGLE_MAPS_API_KEY
    if not api_key:
        return None
    if not _is_under_limit():
        return None

    # Fetch from Google Street View Static API
    try:
        resp = requests.get(STREETVIEW_API_URL, params={
            'size': '600x400',
            'location': f'{lat},{lng}',
            'key': api_key,
            'return_error_codes': 'true',
        }, timeout=10)

        _increment_counter()

        # Google returns 404 or very small images for no-coverage locations
        if resp.status_code != 200 or len(resp.content) < 5000:
            # Save marker so we don't re-request
            with open(_noimage_path(image_hash), 'w') as f:
                pass
            return None

        # Save image to cache
        with open(cached, 'wb') as f:
            f.write(resp.content)
        return cached

    except requests.RequestException:
        return None


def get_streetview_url(lat, lng):
    """Return the proxy URL for a Street View image, or None.

    Does NOT fetch — just returns the URL. The actual fetch happens
    lazily when the browser requests the image from the route.
    Returns None only if there's no API key or the location was
    previously marked as having no coverage.
    """
    if not lat or not lng:
        return None
    if not config.GOOGLE_MAPS_API_KEY:
        return None
    image_hash = _make_hash(lat, lng)
    # Skip locations already known to have no coverage
    if os.path.isfile(_noimage_path(image_hash)):
        return None
    # Include lat/lng so the route can fetch on first request
    return f'/api/streetview/{image_hash}.jpg?lat={lat:.6f}&lng={lng:.6f}'
