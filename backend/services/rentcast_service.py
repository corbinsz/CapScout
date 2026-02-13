"""
RentCast API client with aggressive disk caching.
Free tier: 50 API calls/month.
"""
import os
import json
import time
import hashlib
import requests
import config


RENTCAST_BASE_URL = 'https://api.rentcast.io/v1'
CACHE_TTL_DAYS = 7


def _cache_path(cache_key):
    return os.path.join(config.DATA_PROCESSED_DIR, f'rentcast_{cache_key}.json')


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
    """Make authenticated RentCast API call with caching."""
    if not config.RENTCAST_API_KEY or config.RENTCAST_API_KEY == 'your_api_key_here':
        return None

    params = params or {}
    cache_key = _make_key(endpoint, params)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    url = f"{RENTCAST_BASE_URL}/{endpoint}"
    headers = {'X-Api-Key': config.RENTCAST_API_KEY, 'Accept': 'application/json'}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _save_cache(cache_key, data)
        return data
    except requests.RequestException as e:
        print(f"RentCast API error: {e}")
        return None


def get_listings_for_sale(city='Austin', state='TX', limit=20):
    """Get properties for sale from RentCast."""
    return _api_call('listings/sale', {
        'city': city,
        'state': state,
        'limit': limit,
        'status': 'Active'
    })


def get_property_details(address):
    """Get details for a specific property."""
    return _api_call('properties', {'address': address})


def get_rent_estimate(address):
    """Get rental estimate for a property."""
    return _api_call('avm/rent/long-term', {'address': address})


def get_market_stats(city='Austin', state='TX'):
    """Get market statistics."""
    return _api_call('markets', {'city': city, 'state': state})


def has_api_key():
    """Check if a valid API key is configured."""
    return bool(config.RENTCAST_API_KEY) and config.RENTCAST_API_KEY != 'your_api_key_here'
