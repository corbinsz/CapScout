"""
RapidAPI "Realty in US" (by APIDojo) client with disk caching.
Provides Realtor.com for-sale listings with photos.
"""
import os
import json
import time
import hashlib
import requests
import config


RAPIDAPI_HOST = 'realty-in-us.p.rapidapi.com'
CACHE_TTL_DAYS = 7


def _cache_path(cache_key):
    return os.path.join(config.DATA_PROCESSED_DIR, f'realtor_{cache_key}.json')


def _get_cached(cache_key):
    path = _cache_path(cache_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            cached = json.load(f)
        if time.time() - cached.get('timestamp', 0) < CACHE_TTL_DAYS * 86400:
            return cached['data']
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _save_cache(cache_key, data):
    path = _cache_path(cache_key)
    with open(path, 'w') as f:
        json.dump({'timestamp': time.time(), 'data': data}, f)


def _make_key(endpoint, params):
    raw = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _api_call(endpoint, params=None):
    """Make authenticated RapidAPI call with caching."""
    if not has_api_key():
        return None

    params = params or {}
    cache_key = _make_key(endpoint, params)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    url = f"https://{RAPIDAPI_HOST}/{endpoint}"
    headers = {
        'X-RapidAPI-Key': config.RAPIDAPI_KEY,
        'X-RapidAPI-Host': RAPIDAPI_HOST,
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        _save_cache(cache_key, data)
        return data
    except requests.RequestException as e:
        print(f"RapidAPI Realtor error: {e}")
        return None


def get_listings_for_sale(city='Austin', state_code='TX', limit=50):
    """Get for-sale properties from Realtor.com via RapidAPI."""
    raw = _api_call('properties/v3/list', {
        'city': city,
        'state_code': state_code,
        'offset': 0,
        'limit': limit,
        'status': ['for_sale'],
        'sort': {'direction': 'desc', 'field': 'list_date'},
    })
    if not raw:
        return []

    results = raw.get('data', {}).get('home_search', {}).get('results', [])
    if not results:
        # Try alternate response shape
        results = raw.get('data', {}).get('results', raw.get('results', []))

    return results


def normalize_listing(prop):
    """Normalize a RapidAPI Realtor listing to our unified shape."""
    location = prop.get('location', {})
    address = location.get('address', {})
    description = prop.get('description', {})
    coord = address.get('coordinate', {})

    # Build full address string
    line = address.get('line', '')
    city = address.get('city', '')
    state = address.get('state_code', '')
    postal = address.get('postal_code', '')
    full_address = f"{line}, {city}, {state} {postal}".strip(', ')

    photo_url = None
    primary_photo = prop.get('primary_photo', {})
    if primary_photo and primary_photo.get('href'):
        photo_url = primary_photo['href']

    return {
        'id': prop.get('property_id', prop.get('listing_id', '')),
        'name': full_address or f"Property in {city}",
        'address': full_address,
        'photo_url': photo_url,
        'price': prop.get('list_price', 0),
        'bedrooms': description.get('beds', 0),
        'bathrooms': description.get('baths', 0),
        'sqft': description.get('sqft', 0),
        'latitude': coord.get('lat', 0),
        'longitude': coord.get('lon', 0),
        'property_type': description.get('type', ''),
        'source': 'realtor',
    }


def has_api_key():
    """Check if a valid RapidAPI key is configured."""
    return bool(config.RAPIDAPI_KEY) and config.RAPIDAPI_KEY != 'your_api_key_here'
