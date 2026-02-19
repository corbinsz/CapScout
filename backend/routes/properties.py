import hashlib
import json
import os
import time
import copy
from flask import Blueprint, request, jsonify
import config
from cities import get_city_or_default
from services.data_service import get_all_listings_summary, get_comparable_listings, search_listings
from services.prediction_service import predict_property, models_available
from services.scoring_service import score_property
from services.rentcast_service import get_listings_for_sale as rentcast_for_sale, has_api_key as rentcast_has_key
from services.realtor_service import (
    get_listings_for_sale as realtor_for_sale,
    normalize_listing as normalize_realtor,
    has_api_key as rapidapi_has_key,
)
from services.redfin_service import get_listings as redfin_listings, has_data as redfin_has_data, _property_image_url
from services.streetview_service import get_streetview_url

STREETVIEW_SCORE_THRESHOLD = 30
_SCORE_CACHE_TTL = 600  # 10 minutes
_score_cache = {}  # key -> (timestamp, scored_batch, total_available)

# ---------------------------------------------------------------------------
# Disk-based score cache — survives cold restarts so first load is instant
# ---------------------------------------------------------------------------
_DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'score_cache')
os.makedirs(_DISK_CACHE_DIR, exist_ok=True)


def _disk_cache_path(key):
    """Stable filename for a cache key tuple."""
    h = hashlib.md5(str(key).encode()).hexdigest()
    return os.path.join(_DISK_CACHE_DIR, f'{h}.json')


def _load_disk_cache(key):
    """Load scored batch from disk if it exists."""
    path = _disk_cache_path(key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data['batch'], data['total']
    except Exception:
        return None


def _save_disk_cache(key, batch, total):
    """Persist scored batch to disk."""
    path = _disk_cache_path(key)
    try:
        with open(path, 'w') as f:
            json.dump({'batch': batch, 'total': total}, f)
    except Exception:
        pass


def invalidate_disk_cache(city_key=None):
    """Remove all disk-cached scores (called after CSV upload or retrain)."""
    for fname in os.listdir(_DISK_CACHE_DIR):
        if fname.endswith('.json'):
            try:
                os.remove(os.path.join(_DISK_CACHE_DIR, fname))
            except Exception:
                pass

# Map for-sale property_type strings to the model's binary is_* flags.
# Keys are lowercased substrings matched against the listing's property_type.
_PROPERTY_TYPE_MAP = {
    'single family': 'is_home',
    'multi-family': 'is_home',
    'mobile': 'is_home',
    'manufactured': 'is_home',
    'condo': 'is_condo',
    'co-op': 'is_condo',
    'townhouse': 'is_townhouse',
    'cabin': 'is_cabin',
    'cottage': 'is_cottage',
}


def _property_type_flags(item):
    """Convert a for-sale property_type string to model binary flags."""
    pt = (item.get('property_type') or '').lower()
    if not pt:
        return {}
    flags = {}
    for substring, flag in _PROPERTY_TYPE_MAP.items():
        if substring in pt:
            flags[flag] = 1
            break
    return flags


def _physical_features(item):
    """Extract physical property features from a listing dict for ML prediction."""
    feats = {}
    sqft = item.get('sqft')
    if sqft and sqft > 0:
        feats['sqft'] = sqft
        beds = item.get('bedrooms', 1) or 1
        feats['sqft_per_bedroom'] = sqft / beds
    year = item.get('year_built')
    if year and year > 1800:
        feats['year_built'] = year
        feats['property_age'] = 2026 - year
    lot = item.get('lot_size')
    if lot and lot > 0:
        feats['lot_sqft'] = lot
        feats['lot_acres'] = lot / 43560.0
    return feats

properties_bp = Blueprint('properties', __name__)


@properties_bp.route('/search')
def search_properties():
    """GET /api/search?q=<query>&city=<city_key>
    Search properties by address/name across Redfin and Airbnb data.
    Returns up to 10 results from each source.
    """
    query = request.args.get('q', '').strip()
    city_key = request.args.get('city', 'austin')

    if len(query) < 2:
        return jsonify({'results': []})

    results = []

    # Search Redfin listings (by address)
    if redfin_has_data(city_key):
        q_lower = query.lower()
        for item in redfin_listings(city_key):
            addr = (item.get('address') or item.get('name') or '').lower()
            if q_lower in addr:
                results.append({
                    'id': item['id'],
                    'name': item.get('name') or item.get('address', ''),
                    'address': item.get('address', ''),
                    'price': item.get('price', 0),
                    'bedrooms': item.get('bedrooms', 0),
                    'bathrooms': item.get('bathrooms', 0),
                    'source': 'redfin',
                })
                if len(results) >= 10:
                    break

    # Search Airbnb listings (by name/neighborhood)
    airbnb_results = search_listings(query, city_key=city_key, limit=10)
    results.extend(airbnb_results)

    return jsonify({'results': results[:20]})


@properties_bp.route('/properties')
def list_properties():
    """
    GET /api/properties
    Paginated, filtered, sorted property list with STR investment scores.
    source: 'forsale' (default) | 'redfin' | 'rentcast' | 'realtor' | 'airbnb'
    'forsale' tries Redfin CSV first, then RentCast, then RapidAPI Realtor, falls back to airbnb.
    """
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    bedrooms = request.args.get('bedrooms', type=int)
    neighborhood = request.args.get('neighborhood', type=str)
    sort_by = request.args.get('sort_by', 'score')
    sort_order = request.args.get('sort_order', 'desc')
    source = request.args.get('source', 'forsale')
    city_key = request.args.get('city', 'austin')
    str_viable = request.args.get('str_viable', '').lower() in ('true', '1', 'yes')

    # Parse optional financial assumption overrides
    mortgage_rate_pct = request.args.get('mortgage_rate', type=float)
    mortgage_term = request.args.get('mortgage_term', type=int)
    down_payment_pct_val = request.args.get('down_payment_pct', type=float)
    fin_overrides = {}
    if mortgage_rate_pct is not None:
        fin_overrides['mortgage_rate'] = mortgage_rate_pct / 100
    if mortgage_term is not None:
        fin_overrides['mortgage_term'] = mortgage_term
    if down_payment_pct_val is not None:
        fin_overrides['down_payment_pct'] = down_payment_pct_val / 100

    fs_args = dict(min_price=min_price, max_price=max_price, bedrooms=bedrooms,
                   sort_by=sort_by, sort_order=sort_order, city_key=city_key,
                   str_viable=str_viable, fin_overrides=fin_overrides)

    # --- For-sale cascade: Redfin CSV -> RentCast -> RapidAPI -> Airbnb fallback ---
    if source == 'forsale':
        if redfin_has_data(city_key):
            result = _forsale_properties('redfin', page, per_page, **fs_args)
            if result['total'] > 0:
                return jsonify(result)
        if rentcast_has_key():
            result = _forsale_properties('rentcast', page, per_page, **fs_args)
            if result['total'] > 0:
                return jsonify(result)
        if rapidapi_has_key():
            result = _forsale_properties('realtor', page, per_page, **fs_args)
            if result['total'] > 0:
                return jsonify(result)
        # Fallback to airbnb with a message
        return _airbnb_properties(page, per_page, min_price, max_price, bedrooms,
                                  neighborhood, sort_by, sort_order, city_key,
                                  message='No for-sale API keys configured. Showing Airbnb data as fallback.')

    if source == 'redfin' and redfin_has_data(city_key):
        return jsonify(_forsale_properties('redfin', page, per_page, **fs_args))

    if source == 'rentcast' and rentcast_has_key():
        return jsonify(_forsale_properties('rentcast', page, per_page, **fs_args))

    if source == 'realtor' and rapidapi_has_key():
        return jsonify(_forsale_properties('realtor', page, per_page, **fs_args))

    # Default: Airbnb
    return _airbnb_properties(page, per_page, min_price, max_price, bedrooms,
                              neighborhood, sort_by, sort_order, city_key)


def _score_item(item, purchase_price, city_key=None, fin_overrides=None):
    """Add ML predictions and investment score to an item dict."""
    if not models_available(city_key):
        return
    city = get_city_or_default(city_key)
    try:
        features = {
            'bedrooms': item.get('bedrooms', 2),
            'bathrooms_clean': item.get('bathrooms', 1),
            'accommodates': item.get('bedrooms', 2) * 2,
            'latitude': item.get('latitude', city['latitude']),
            'longitude': item.get('longitude', city['longitude']),
            **_physical_features(item),
            **_property_type_flags(item),
        }
        prediction = predict_property(features, city_key=city_key, summary_only=True)
        hoa_annual = (item.get('hoa_monthly') or 0) * 12
        scoring = score_property(
            purchase_price,
            prediction['predicted_annual_revenue'],
            prediction['predicted_occupancy_rate'],
            city_key=city_key,
            hoa_annual=hoa_annual,
            **(fin_overrides or {}),
        )
        item['score'] = scoring['score']
        item['predicted'] = prediction
        item['financials'] = scoring['financials']
        item['score_breakdown'] = scoring.get('breakdown')
        conf = prediction.get('confidence') or {}
        item['confidence_level'] = conf.get('level', 'medium')
    except Exception:
        item['score'] = None
        item['predicted'] = None
        item['financials'] = None
        item['confidence_level'] = 'low'


def _cache_key(backend, city_key, min_price, max_price, bedrooms, fin_overrides):
    """Build a hashable cache key for scored batches."""
    fin = tuple(sorted((fin_overrides or {}).items()))
    return (backend, city_key, min_price, max_price, bedrooms, fin)


def _get_scored_batch(backend, city_key, min_price, max_price, bedrooms, fin_overrides):
    """Return scored batch from memory cache, disk cache, or compute + cache."""
    key = _cache_key(backend, city_key, min_price, max_price, bedrooms, fin_overrides)
    now = time.time()

    # 1. Memory cache (fastest)
    cached = _score_cache.get(key)
    if cached and (now - cached[0]) < _SCORE_CACHE_TTL:
        return copy.deepcopy(cached[1]), cached[2]

    # 2. Disk cache (survives cold restarts)
    disk = _load_disk_cache(key)
    if disk:
        batch, total = disk
        _score_cache[key] = (now, copy.deepcopy(batch), total)
        return batch, total

    # 3. Compute from scratch
    city = get_city_or_default(city_key)
    if backend == 'redfin':
        items = redfin_listings(city_key)
    elif backend == 'rentcast':
        raw = rentcast_for_sale(city=city['name'], state=city['state'], limit=50)
        items = _normalize_rentcast_list(raw or [])
    else:
        raw = realtor_for_sale(city=city['name'], state_code=city['state'], limit=50)
        items = [normalize_realtor(p) for p in (raw or [])]

    filtered = []
    for item in items:
        price = item.get('price', 0)
        if not price or price <= 0:
            continue
        if bedrooms and item.get('bedrooms') != bedrooms:
            continue
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue
        filtered.append(item)

    total_available = len(filtered)

    SCORE_BATCH = 100
    filtered.sort(key=lambda x: x.get('price', 0))
    batch = filtered[:SCORE_BATCH]
    for item in batch:
        _score_item(item, item.get('price', 0), city_key, fin_overrides)

    _score_cache[key] = (now, copy.deepcopy(batch), total_available)
    _save_disk_cache(key, batch, total_available)
    return batch, total_available


def _forsale_properties(backend, page, per_page, min_price, max_price,
                        bedrooms, sort_by, sort_order, city_key=None,
                        str_viable=False, fin_overrides=None):
    """Build unified property list from a for-sale backend."""
    if sort_by == 'score':
        batch, total_available = _get_scored_batch(
            backend, city_key, min_price, max_price, bedrooms, fin_overrides)
        if str_viable:
            batch = [it for it in batch if it.get('confidence_level') != 'low']
        batch.sort(key=lambda x: x.get('score') or 0, reverse=(sort_order == 'desc'))
        start = (page - 1) * per_page
        page_items = batch[start:start + per_page]
        total = len(batch)
    else:
        # Price sort: paginate first, then score just the page
        batch, total_available = _get_scored_batch(
            backend, city_key, min_price, max_price, bedrooms, fin_overrides)
        batch.sort(key=lambda x: x.get('price', 0), reverse=(sort_order == 'desc'))
        total = len(batch)
        start = (page - 1) * per_page
        page_items = batch[start:start + per_page]
        if str_viable:
            page_items = [it for it in page_items if it.get('confidence_level') != 'low']

    # Upgrade photo_url to Street View for high-scoring properties
    for item in page_items:
        if (item.get('score') or 0) >= STREETVIEW_SCORE_THRESHOLD:
            sv_url = get_streetview_url(item.get('latitude'), item.get('longitude'))
            if sv_url:
                item['photo_url'] = sv_url

    return {
        'items': page_items,
        'total': total,
        'total_available': total_available,
        'page': page,
        'per_page': per_page,
    }


def _normalize_rentcast_list(raw):
    """Normalize RentCast listings to unified shape."""
    if not isinstance(raw, list):
        return []
    items = []
    for prop in raw:
        items.append({
            'id': prop.get('id', prop.get('formattedAddress', '')),
            'name': prop.get('formattedAddress', ''),
            'address': prop.get('formattedAddress', ''),
            'photo_url': _property_image_url(
                prop.get('latitude'), prop.get('longitude'),
                prop.get('formattedAddress', '')),
            'price': prop.get('price', 0),
            'bedrooms': prop.get('bedrooms', 0),
            'bathrooms': prop.get('bathrooms', 0),
            'sqft': prop.get('squareFootage', 0),
            'latitude': prop.get('latitude', 0),
            'longitude': prop.get('longitude', 0),
            'property_type': prop.get('propertyType', ''),
            'source': 'rentcast',
        })
    return items


def _airbnb_properties(page, per_page, min_price, max_price, bedrooms,
                       neighborhood, sort_by, sort_order, city_key=None, message=None):
    """Use Inside Airbnb data with ML predictions."""
    city = get_city_or_default(city_key)
    airbnb_sort = 'price_clean' if sort_by in ('price', 'score') else sort_by
    result = get_all_listings_summary(
        page=page, per_page=per_page,
        min_price=min_price, max_price=max_price,
        bedrooms=bedrooms, neighborhood=neighborhood,
        sort_by=airbnb_sort, sort_order=sort_order,
        city_key=city_key,
    )

    # Airbnb listings show actual performance — no fake investment scoring
    if sort_by == 'score':
        # Default sort for Airbnb: by monthly revenue descending
        result['items'].sort(key=lambda x: x.get('monthly_revenue') or 0, reverse=(sort_order == 'desc'))

    # Add source marker and normalize fields for airbnb items
    for item in result['items']:
        item['source'] = 'airbnb'
        item.setdefault('address', '')
        item.setdefault('photo_url', None)

    if message:
        result['message'] = message

    return jsonify(result)


@properties_bp.route('/properties/<property_id>')
def get_property(property_id):
    """GET /api/properties/:id — full detail + comparables + neighborhood stats."""
    from services.data_service import get_listing_by_id, get_comparable_listings, get_neighborhood_stats

    city_key = request.args.get('city', 'austin')
    city = get_city_or_default(city_key)

    # Parse optional financial assumption overrides
    mortgage_rate_pct = request.args.get('mortgage_rate', type=float)
    mortgage_term = request.args.get('mortgage_term', type=int)
    down_payment_pct_val = request.args.get('down_payment_pct', type=float)
    # Convert from user-friendly % to decimal (e.g. 7 -> 0.07)
    fin_overrides = {}
    if mortgage_rate_pct is not None:
        fin_overrides['mortgage_rate'] = mortgage_rate_pct / 100
    if mortgage_term is not None:
        fin_overrides['mortgage_term'] = mortgage_term
    if down_payment_pct_val is not None:
        fin_overrides['down_payment_pct'] = down_payment_pct_val / 100

    # Check Redfin listings first if the ID looks like a Redfin ID
    target = None
    if property_id.startswith('redfin-'):
        listings = redfin_listings(city_key)
        target = next((l for l in listings if l['id'] == property_id), None)

    if not target:
        target = get_listing_by_id(property_id, city_key=city_key)

    if not target:
        return jsonify({'error': 'Property not found'}), 404

    is_forsale = target.get('source') in ('redfin', 'rentcast', 'realtor')

    # Add predictions
    if models_available(city_key):
        try:
            prediction = predict_property({
                'bedrooms': target.get('bedrooms', 1),
                'bathrooms_clean': target.get('bathrooms', 1),
                'accommodates': target.get('accommodates', target.get('bedrooms', 2) * 2),
                'latitude': target.get('latitude', city['latitude']),
                'longitude': target.get('longitude', city['longitude']),
                **_physical_features(target),
                **_property_type_flags(target),
            }, city_key=city_key)
            target['predicted'] = prediction
            if is_forsale:
                purchase_price = target['price']
                hoa_annual = (target.get('hoa_monthly') or 0) * 12
                scoring = score_property(purchase_price, prediction['predicted_annual_revenue'],
                                          prediction['predicted_occupancy_rate'],
                                          city_key=city_key, hoa_annual=hoa_annual,
                                          **fin_overrides)
                target['score'] = scoring['score']
                target['score_breakdown'] = scoring['breakdown']
                target['financials'] = scoring['financials']
        except Exception:
            pass

    # Upgrade to Street View photo when viewing a single property
    if is_forsale and target.get('latitude') and target.get('longitude'):
        sv_url = get_streetview_url(target['latitude'], target['longitude'])
        if sv_url:
            target['photo_url'] = sv_url

    target.setdefault('source', 'airbnb')
    target.setdefault('address', '')
    target.setdefault('photo_url', None)

    # Find comparable listings
    comps = get_comparable_listings(
        target['latitude'], target['longitude'],
        target.get('bedrooms'), radius_km=3, limit=10, city_key=city_key
    )

    # Neighborhood stats
    neighborhoods = get_neighborhood_stats(city_key)
    target_hood = target.get('neighbourhood', '')
    hood_stats = next((n for n in neighborhoods
                       if n['neighbourhood_cleansed'] == target_hood), None)

    return jsonify({
        'property': target,
        'comparables': comps,
        'neighborhood_stats': hood_stats,
    })
