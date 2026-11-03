"""Licensed for-sale listings from the RentCast API.

The API key is read from RENTCAST_API_KEY. Responses are cached on disk so a
page view does not create a billed request. Nothing in this module calls
Redfin.
"""
import json
import time

import requests

import config
from cities import get_city
from services.providers.base import empty_listing_message
from services.providers.normalize import listing_id, osm_tile_url, safe_float, safe_int

RENTCAST_BASE_URL = 'https://api.rentcast.io/v1'
_EXCLUDED_TYPES = {'land', 'vacant land'}


class RentCastProvider:
    name = 'rentcast'
    label = 'RentCast'
    licensed = True
    notice = 'For-sale listings from RentCast, a licensed property-data API.'

    def empty_message(self, city_key):
        return empty_listing_message(self.name)

    def listings(self, city_key):
        cached = _read_cache(city_key)
        if cached is not None:
            return cached
        if not config.RENTCAST_API_KEY:
            return []
        raw_rows = _fetch_city(city_key)
        if raw_rows is None:
            # Do not cache a transport failure; the next request may succeed.
            return []
        normalized = [row for row in (normalize_rentcast(item) for item in raw_rows) if row]
        _write_cache(city_key, normalized)
        return normalized

    def get(self, city_key, property_id):
        if not str(property_id).startswith('rentcast-'):
            return None
        for item in self.listings(city_key):
            if item['id'] == property_id:
                return item
        return None


def normalize_rentcast(prop):
    """Map a RentCast sale-listing object onto the unified listing dict."""
    if not isinstance(prop, dict):
        return None
    status = str(prop.get('status') or 'Active').strip().lower()
    if status and status not in {'active', 'for sale', 'for_sale'}:
        return None
    prop_type = str(prop.get('propertyType') or prop.get('property_type') or '')
    if prop_type.strip().lower() in _EXCLUDED_TYPES:
        return None

    address = (
        prop.get('formattedAddress')
        or prop.get('addressLine1')
        or prop.get('address')
        or ''
    )
    address = str(address).strip()
    price = safe_float(prop.get('price'))
    bedrooms = safe_int(prop.get('bedrooms'))
    if not address or price <= 0 or bedrooms < 1:
        return None

    raw_key = prop.get('id') or address
    lat = safe_float(prop.get('latitude'))
    lng = safe_float(prop.get('longitude'))
    lot = safe_int(prop.get('lotSize') or prop.get('lot_size'))
    if lot > 217800:
        lot = 0
    hoa = prop.get('hoa') or {}
    if isinstance(hoa, dict):
        hoa_monthly = safe_float(hoa.get('fee'))
    else:
        hoa_monthly = safe_float(hoa)

    return {
        'id': listing_id('rentcast', raw_key),
        'provider_id': str(raw_key),
        'name': address,
        'address': address,
        'photo_url': osm_tile_url(lat, lng),
        'price': price,
        'bedrooms': bedrooms,
        'bathrooms': safe_float(prop.get('bathrooms')),
        'sqft': safe_int(prop.get('squareFootage') or prop.get('sqft')),
        'year_built': safe_int(prop.get('yearBuilt') or prop.get('year_built')),
        'lot_size': lot,
        'days_on_market': safe_int(prop.get('daysOnMarket') or prop.get('days_on_market')),
        'hoa_monthly': hoa_monthly,
        'price_per_sqft': safe_float(prop.get('pricePerSquareFoot')),
        'latitude': lat,
        'longitude': lng,
        'property_type': prop_type,
        'url': '',
        'source': 'rentcast',
    }


def _cache_path(city_key):
    return os_path(city_key)


def os_path(city_key):
    import os
    return os.path.join(config.cache_dir(), f'rentcast_{city_key}.json')


def _read_cache(city_key):
    import os
    path = os_path(city_key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        age = time.time() - float(payload.get('saved_at') or 0)
        if age > config.RENTCAST_CACHE_HOURS * 3600:
            return None
        rows = payload.get('listings')
        if isinstance(rows, list):
            return rows
    except (OSError, ValueError, TypeError):
        return None
    return None


def _write_cache(city_key, listings):
    path = os_path(city_key)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump({'saved_at': time.time(), 'listings': listings}, handle)


def _fetch_city(city_key):
    city = get_city(city_key)
    headers = {
        'X-Api-Key': config.RENTCAST_API_KEY,
        'Accept': 'application/json',
    }
    rows = []
    for page in range(config.RENTCAST_MAX_PAGES):
        params = {
            'city': city['name'],
            'state': city['state'],
            'status': 'Active',
            'limit': 500,
            'offset': page * 500,
        }
        try:
            response = requests.get(
                f'{RENTCAST_BASE_URL}/listings/sale',
                headers=headers,
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f'[rentcast] request failed for {city_key}: {exc.__class__.__name__}')
            return None if page == 0 else rows
        if isinstance(payload, dict):
            print(f'[rentcast] unexpected payload for {city_key}')
            return None if page == 0 else rows
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        if len(payload) < 500:
            break
    return rows
