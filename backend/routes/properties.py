import hashlib
import json
import os
import time
import copy
from flask import Blueprint, request, jsonify
import config
from cities import get_city_or_default
from services.data_service import get_all_listings_summary, search_listings
from services.prediction_service import predict_property, predict_batch, models_available
from services.scoring_service import explain_deal, score_property
from services.providers.registry import active_provider, provider_status
from services.streetview_service import get_streetview_url
from security import bounded_float, bounded_int, city_or_400, clean_text, rate_limit

STREETVIEW_SCORE_THRESHOLD = 30
_SCORE_CACHE_TTL = 600  # 10 minutes in memory
_DISK_CACHE_TTL = 6 * 3600
_SCORE_CACHE_VERSION = 5  # Bump when scoring, insight copy, or listing shape changes
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
        saved_at = data.get('saved_at')
        if saved_at is None or (time.time() - float(saved_at)) > _DISK_CACHE_TTL:
            return None
        return data['batch'], data['total']
    except Exception:
        return None


def _save_disk_cache(key, batch, total):
    """Persist scored batch to disk."""
    path = _disk_cache_path(key)
    try:
        with open(path, 'w') as f:
            json.dump({'saved_at': time.time(), 'batch': batch, 'total': total}, f)
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
@rate_limit('search', limit=60, window=60)
def search_properties():
    """GET /api/search?q=<query>&city=<city_key>
    Search the active for-sale provider and Inside Airbnb comps.
    """
    query = clean_text(request.args.get('q', ''), 80)
    city_key, error = city_or_400(request.args.get('city'))
    if error:
        return error

    if len(query) < 2:
        return jsonify({'results': []})

    results = []
    provider = active_provider()
    if provider is not None:
        q_lower = query.lower()
        for item in provider.listings(city_key):
            addr = (item.get('address') or item.get('name') or '').lower()
            if q_lower in addr:
                results.append({
                    'id': item['id'],
                    'name': item.get('name') or item.get('address', ''),
                    'address': item.get('address', ''),
                    'price': item.get('price', 0),
                    'bedrooms': item.get('bedrooms', 0),
                    'bathrooms': item.get('bathrooms', 0),
                    'source': item.get('source') or provider.name,
                })
                if len([row for row in results if row.get('source') != 'airbnb']) >= 10:
                    break

    results.extend(search_listings(query, city_key=city_key, limit=10))
    return jsonify({'results': results[:20]})


def _parse_listing_filters():
    """Shared, bounded query params for the list and the paid report."""
    try:
        page = bounded_int(request.args.get('page'), 1, 10000) or 1
        per_page = bounded_int(request.args.get('per_page'), 1, 50) or 20
        min_price = bounded_float(request.args.get('min_price'), 0, 100_000_000)
        max_price = bounded_float(request.args.get('max_price'), 0, 100_000_000)
        bedrooms = bounded_int(request.args.get('bedrooms'), 0, 20)
        mortgage_rate_pct = bounded_float(request.args.get('mortgage_rate'), 0, 25)
        mortgage_term = bounded_int(request.args.get('mortgage_term'), 1, 40)
        down_payment_pct_val = bounded_float(request.args.get('down_payment_pct'), 0, 100)
    except ValueError:
        return None, (jsonify({'error': 'A filter value is out of range.'}), 400)

    sort_by = request.args.get('sort_by', 'score')
    if sort_by not in {'score', 'price'}:
        sort_by = 'score'
    sort_order = request.args.get('sort_order', 'desc')
    if sort_order not in {'asc', 'desc'}:
        sort_order = 'desc'
    property_type = clean_text(request.args.get('property_type'), 40) or None
    neighborhood = clean_text(request.args.get('neighborhood'), 80) or None
    fin_overrides = {}
    if mortgage_rate_pct is not None:
        fin_overrides['mortgage_rate'] = mortgage_rate_pct / 100
    if mortgage_term is not None:
        fin_overrides['mortgage_term'] = mortgage_term
    if down_payment_pct_val is not None:
        fin_overrides['down_payment_pct'] = down_payment_pct_val / 100
    return {
        'page': page,
        'per_page': per_page,
        'min_price': min_price,
        'max_price': max_price,
        'bedrooms': bedrooms,
        'neighborhood': neighborhood,
        'sort_by': sort_by,
        'sort_order': sort_order,
        'property_type': property_type,
        'str_viable': request.args.get('str_viable', '').lower() in ('true', '1', 'yes'),
        'source': request.args.get('source', 'forsale'),
        'fin_overrides': fin_overrides,
    }, None


@properties_bp.route('/properties')
@rate_limit('properties', limit=90, window=60)
def list_properties():
    """GET /api/properties — scored for-sale homes from the active provider."""
    city_key, error = city_or_400(request.args.get('city'))
    if error:
        return error
    filters, error = _parse_listing_filters()
    if error:
        return error

    status = provider_status()
    source = filters.pop('source')
    if source == 'airbnb':
        result = _airbnb_properties(city_key=city_key, **filters)
        result['data_source'] = 'inside_airbnb'
        result['data_source_label'] = 'Inside Airbnb'
        return jsonify(result)

    if status['source'] is None:
        result = _airbnb_properties(
            city_key=city_key,
            message=status['message'],
            **filters,
        )
        result['data_source'] = None
        result['data_source_label'] = None
        return jsonify(result)

    payload = _forsale_properties(city_key=city_key, **filters)
    payload['data_source'] = status['source']
    payload['data_source_label'] = status['label']
    payload['data_licensed'] = status['licensed']
    if payload['total'] == 0:
        payload['message'] = active_provider().empty_message(city_key)
    else:
        payload['message'] = status['message']
    return jsonify(payload)


def _model_token(city_key):
    stamps = []
    for name in ('price_model.joblib', 'occupancy_model.joblib'):
        path = os.path.join(config.city_models_dir(city_key), name)
        try:
            stamps.append(int(os.path.getmtime(path)))
        except OSError:
            stamps.append(0)
    return tuple(stamps)


def _features_for(item, city):
    bedrooms = item.get('bedrooms') or 2
    return {
        'bedrooms': bedrooms,
        'bathrooms_clean': item.get('bathrooms') or 1,
        'accommodates': item.get('accommodates') or bedrooms * 2,
        'latitude': item.get('latitude') or city['latitude'],
        'longitude': item.get('longitude') or city['longitude'],
        **_physical_features(item),
        **_property_type_flags(item),
    }


def _attach_scores(items, city_key, fin_overrides):
    """Score every listing in one model call so rank is not biased to cheap homes."""
    if not items or not models_available(city_key):
        return
    city = get_city_or_default(city_key)
    try:
        predictions = predict_batch([_features_for(item, city) for item in items], city_key=city_key)
    except Exception:
        for item in items:
            item['score'] = None
            item['predicted'] = None
            item['financials'] = None
            item['confidence_level'] = 'low'
            item['explanation'] = explain_deal(None, None)
        return
    for item, prediction in zip(items, predictions):
        hoa_annual = (item.get('hoa_monthly') or 0) * 12
        scoring = score_property(
            item.get('price') or 0,
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
        level = (prediction.get('confidence') or {}).get('level', 'medium')
        item['confidence_level'] = level
        item['explanation'] = explain_deal(scoring['score'], scoring['financials'], level)


def _cache_key(source, city_key, min_price, max_price, bedrooms, fin_overrides, property_type=None):
    fin = tuple(sorted((fin_overrides or {}).items()))
    return (
        _SCORE_CACHE_VERSION, source, city_key, min_price, max_price,
        bedrooms, property_type, fin, _model_token(city_key),
    )


def _filter_listings(items, min_price, max_price, bedrooms, property_type):
    filtered = []
    for item in items:
        price = item.get('price') or 0
        if price <= 0:
            continue
        if bedrooms and item.get('bedrooms') != bedrooms:
            continue
        if property_type and property_type.lower() not in (item.get('property_type') or '').lower():
            continue
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue
        filtered.append(item)
    return filtered


def score_city_listings(city_key, min_price=None, max_price=None, bedrooms=None,
                        property_type=None, fin_overrides=None):
    """Score the active provider's listings. Shared by the list API and the report."""
    provider = active_provider()
    status = provider_status()
    if provider is None:
        return [], status
    key = _cache_key(provider.name, city_key, min_price, max_price, bedrooms, fin_overrides, property_type)
    now = time.time()
    cached = _score_cache.get(key)
    if cached and (now - cached[0]) < _SCORE_CACHE_TTL:
        return copy.deepcopy(cached[1]), status
    disk = _load_disk_cache(key)
    if disk:
        batch, _total = disk
        _score_cache[key] = (now, copy.deepcopy(batch), len(batch))
        return batch, status

    filtered = _filter_listings(
        provider.listings(city_key), min_price, max_price, bedrooms, property_type,
    )
    _attach_scores(filtered, city_key, fin_overrides)
    _score_cache[key] = (now, copy.deepcopy(filtered), len(filtered))
    _save_disk_cache(key, filtered, len(filtered))
    return filtered, status


def _forsale_properties(page, per_page, min_price, max_price, bedrooms, sort_by,
                        sort_order, city_key=None, str_viable=False, fin_overrides=None,
                        property_type=None, neighborhood=None):
    """Rank every filtered home, then paginate. Confidence filter happens first."""
    del neighborhood  # for-sale rows have no Inside Airbnb neighbourhood field
    batch, _status = score_city_listings(
        city_key, min_price, max_price, bedrooms, property_type, fin_overrides,
    )
    total_available = len(batch)
    if str_viable:
        batch = [item for item in batch if item.get('confidence_level') != 'low']
    if sort_by == 'price':
        batch.sort(key=lambda item: item.get('price') or 0, reverse=(sort_order == 'desc'))
    else:
        batch.sort(key=lambda item: item.get('score') if item.get('score') is not None else -1,
                   reverse=(sort_order == 'desc'))
    total = len(batch)
    _annotate_rank(batch)
    start = (page - 1) * per_page
    page_items = batch[start:start + per_page]
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


def _annotate_rank(batch):
    """Say how a home compares with the other homes in this search."""
    ranked = sorted(
        [item for item in batch if item.get('score') is not None],
        key=lambda item: item['score'],
        reverse=True,
    )
    total = len(ranked)
    if total < 5:
        return
    for index, item in enumerate(ranked):
        beats = min(99, int(round(100 * (total - index - 1) / total)))
        item['beats_percent'] = beats
        score = item.get('score') or 0
        if beats >= 90 and score >= 55:
            item['insight'] = f'Beats about {beats}% of the homes in this search.'
        elif beats >= 90:
            item['insight'] = (
                f'Higher than about {beats}% of this search, and still short of covering a typical mortgage.'
            )
        elif beats >= 80 and score >= 55:
            item['insight'] = f'Stronger than about {beats}% of the homes in this search.'


def _airbnb_properties(page, per_page, min_price, max_price, bedrooms,
                       neighborhood, sort_by, sort_order, city_key=None, message=None,
                       **_ignored):
    """Inside Airbnb comps. These are active rentals, not homes for sale."""
    airbnb_sort = 'price_clean' if sort_by in ('price', 'score') else sort_by
    result = get_all_listings_summary(
        page=page, per_page=per_page,
        min_price=min_price, max_price=max_price,
        bedrooms=bedrooms, neighborhood=neighborhood,
        sort_by=airbnb_sort, sort_order=sort_order,
        city_key=city_key,
    )
    if sort_by == 'score':
        result['items'].sort(key=lambda x: x.get('monthly_revenue') or 0, reverse=(sort_order == 'desc'))
    for item in result['items']:
        item['source'] = 'airbnb'
        item.setdefault('address', '')
        item.setdefault('photo_url', None)
        item['explanation'] = 'Active short-term rental from Inside Airbnb, not a home for sale.'
    if message:
        result['message'] = message
    return result


@properties_bp.route('/properties/<property_id>')
@rate_limit('property', limit=120, window=60)
def get_property(property_id):
    """GET /api/properties/:id — full detail + comparables + neighborhood stats."""
    from services.data_service import get_listing_by_id, get_comparable_listings, get_neighborhood_stats
    from security import valid_property_id

    if not valid_property_id(property_id):
        return jsonify({'error': 'Property not found'}), 404

    city_key, error = city_or_400(request.args.get('city'))
    if error:
        return error
    city = get_city_or_default(city_key)

    try:
        mortgage_rate_pct = bounded_float(request.args.get('mortgage_rate'), 0, 25)
        mortgage_term = bounded_int(request.args.get('mortgage_term'), 1, 40)
        down_payment_pct_val = bounded_float(request.args.get('down_payment_pct'), 0, 100)
    except ValueError:
        return jsonify({'error': 'A financing value is out of range.'}), 400
    # Convert from user-friendly % to decimal (e.g. 7 -> 0.07)
    fin_overrides = {}
    if mortgage_rate_pct is not None:
        fin_overrides['mortgage_rate'] = mortgage_rate_pct / 100
    if mortgage_term is not None:
        fin_overrides['mortgage_term'] = mortgage_term
    if down_payment_pct_val is not None:
        fin_overrides['down_payment_pct'] = down_payment_pct_val / 100

    target = None
    provider = active_provider()
    if provider is not None:
        target = provider.get(city_key, property_id)

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
                level = (prediction.get('confidence') or {}).get('level')
                target['confidence_level'] = level
                target['explanation'] = explain_deal(scoring['score'], scoring['financials'], level)
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


# Feature name labels for human-readable display
_FEATURE_LABELS = {
    'bedrooms': 'Bedrooms',
    'bathrooms_clean': 'Bathrooms',
    'accommodates': 'Guest Capacity',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'sqft': 'Square Footage',
    'sqft_per_bedroom': 'Sqft per Bedroom',
    'year_built': 'Year Built',
    'property_age': 'Property Age',
    'lot_sqft': 'Lot Size (sqft)',
    'lot_acres': 'Lot Size (acres)',
    'is_home': 'Single Family Home',
    'is_condo': 'Condo/Co-op',
    'is_townhouse': 'Townhouse',
    'is_cabin': 'Cabin',
    'is_cottage': 'Cottage',
    'cluster_label': 'Neighborhood Cluster',
    'month': 'Month',
}


@properties_bp.route('/properties/<property_id>/feature-importance')
def feature_importance(property_id):
    """GET /api/properties/<id>/feature-importance?city=<city_key>
    Returns top feature importances from the XGBoost models."""
    from services.prediction_service import get_models

    city_key = request.args.get('city', 'austin')
    models = get_models(city_key)
    if not models:
        return jsonify({'error': 'No models available'}), 404

    result = {}
    for model_key, label in [('price_model', 'Price Drivers'), ('occupancy_model', 'Occupancy Drivers')]:
        model = models.get(model_key)
        if model is None:
            continue
        try:
            importances = model.feature_importances_
            feature_names = model.get_booster().feature_names
            if not feature_names:
                continue
            pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
            result[model_key.replace('_model', '')] = {
                'label': label,
                'features': [
                    {'name': _FEATURE_LABELS.get(name, name), 'importance': round(float(imp), 4)}
                    for name, imp in pairs
                ],
            }
        except Exception:
            continue

    return jsonify(result)
