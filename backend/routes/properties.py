from flask import Blueprint, request, jsonify
from services.data_service import get_all_listings_summary, get_comparable_listings
from services.prediction_service import predict_property, models_available
from services.scoring_service import score_property
from services.rentcast_service import get_listings_for_sale as rentcast_for_sale, has_api_key as rentcast_has_key
from services.realtor_service import (
    get_listings_for_sale as realtor_for_sale,
    normalize_listing as normalize_realtor,
    has_api_key as rapidapi_has_key,
)
from services.redfin_service import get_listings as redfin_listings, has_data as redfin_has_data

properties_bp = Blueprint('properties', __name__)


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

    # --- For-sale cascade: Redfin CSV -> RentCast -> RapidAPI -> Airbnb fallback ---
    if source == 'forsale':
        if redfin_has_data():
            result = _forsale_properties('redfin', page, per_page, min_price, max_price,
                                         bedrooms, sort_by, sort_order)
            if result['total'] > 0:
                return jsonify(result)
        if rentcast_has_key():
            result = _forsale_properties('rentcast', page, per_page, min_price, max_price,
                                         bedrooms, sort_by, sort_order)
            if result['total'] > 0:
                return jsonify(result)
        if rapidapi_has_key():
            result = _forsale_properties('realtor', page, per_page, min_price, max_price,
                                         bedrooms, sort_by, sort_order)
            if result['total'] > 0:
                return jsonify(result)
        # Fallback to airbnb with a message
        return _airbnb_properties(page, per_page, min_price, max_price, bedrooms,
                                  neighborhood, sort_by, sort_order,
                                  message='No for-sale API keys configured. Showing Airbnb data as fallback.')

    if source == 'redfin' and redfin_has_data():
        return jsonify(_forsale_properties('redfin', page, per_page, min_price, max_price,
                                           bedrooms, sort_by, sort_order))

    if source == 'rentcast' and rentcast_has_key():
        return jsonify(_forsale_properties('rentcast', page, per_page, min_price, max_price,
                                           bedrooms, sort_by, sort_order))

    if source == 'realtor' and rapidapi_has_key():
        return jsonify(_forsale_properties('realtor', page, per_page, min_price, max_price,
                                           bedrooms, sort_by, sort_order))

    # Default: Airbnb
    return _airbnb_properties(page, per_page, min_price, max_price, bedrooms,
                              neighborhood, sort_by, sort_order)


def _score_item(item, purchase_price):
    """Add ML predictions and investment score to an item dict."""
    if not models_available():
        return
    try:
        features = {
            'bedrooms': item.get('bedrooms', 2),
            'bathrooms_clean': item.get('bathrooms', 1),
            'accommodates': item.get('bedrooms', 2) * 2,
            'latitude': item.get('latitude', 30.27),
            'longitude': item.get('longitude', -97.74),
        }
        prediction = predict_property(features)
        scoring = score_property(
            purchase_price,
            prediction['predicted_annual_revenue'],
            prediction['predicted_occupancy_rate'],
        )
        item['score'] = scoring['score']
        item['predicted'] = prediction
        item['financials'] = scoring['financials']
        item['score_breakdown'] = scoring.get('breakdown')
    except Exception:
        item['score'] = None
        item['predicted'] = None
        item['financials'] = None


def _forsale_properties(backend, page, per_page, min_price, max_price,
                        bedrooms, sort_by, sort_order):
    """Build unified property list from a for-sale backend."""
    if backend == 'redfin':
        items = redfin_listings()
    elif backend == 'rentcast':
        raw = rentcast_for_sale(limit=50)
        items = _normalize_rentcast_list(raw or [])
    else:
        raw = realtor_for_sale(limit=50)
        items = [normalize_realtor(p) for p in (raw or [])]

    # Filter
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
        _score_item(item, price)
        filtered.append(item)

    # Sort
    if sort_by == 'score':
        filtered.sort(key=lambda x: x.get('score') or 0, reverse=(sort_order == 'desc'))
    elif sort_by == 'price':
        filtered.sort(key=lambda x: x.get('price', 0), reverse=(sort_order == 'desc'))

    total = len(filtered)
    start = (page - 1) * per_page
    return {
        'items': filtered[start:start + per_page],
        'total': total,
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
            'photo_url': None,  # RentCast doesn't provide photos
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
                       neighborhood, sort_by, sort_order, message=None):
    """Use Inside Airbnb data with ML predictions."""
    airbnb_sort = 'price_clean' if sort_by == 'price' else 'price_clean'
    result = get_all_listings_summary(
        page=page, per_page=per_page,
        min_price=min_price, max_price=max_price,
        bedrooms=bedrooms, neighborhood=neighborhood,
        sort_by=airbnb_sort, sort_order=sort_order,
    )

    if models_available():
        for item in result['items']:
            try:
                prediction = predict_property({
                    'bedrooms': item.get('bedrooms', 1),
                    'bathrooms_clean': item.get('bathrooms', 1),
                    'accommodates': item.get('accommodates', 2),
                    'latitude': item.get('latitude', 30.27),
                    'longitude': item.get('longitude', -97.74),
                })
                proxy_purchase = item['price'] * 2000
                scoring = score_property(
                    proxy_purchase,
                    prediction['predicted_annual_revenue'],
                    prediction['predicted_occupancy_rate'],
                )
                item['score'] = scoring['score']
                item['predicted'] = prediction
                item['financials'] = scoring['financials']
            except Exception:
                item['score'] = None
                item['predicted'] = None

        if sort_by == 'score':
            scored = [i for i in result['items'] if i.get('score') is not None]
            unscored = [i for i in result['items'] if i.get('score') is None]
            scored.sort(key=lambda x: x['score'], reverse=(sort_order == 'desc'))
            result['items'] = scored + unscored

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

    # Check Redfin listings first if the ID looks like a Redfin ID
    target = None
    if property_id.startswith('redfin-'):
        listings = redfin_listings()
        target = next((l for l in listings if l['id'] == property_id), None)

    if not target:
        target = get_listing_by_id(property_id)

    if not target:
        return jsonify({'error': 'Property not found'}), 404

    is_forsale = target.get('source') in ('redfin', 'rentcast', 'realtor')

    # Add predictions
    if models_available():
        try:
            purchase_price = target['price'] if is_forsale else target['price'] * 2000
            prediction = predict_property({
                'bedrooms': target.get('bedrooms', 1),
                'bathrooms_clean': target.get('bathrooms', 1),
                'accommodates': target.get('accommodates', target.get('bedrooms', 2) * 2),
                'latitude': target.get('latitude', 30.27),
                'longitude': target.get('longitude', -97.74),
            })
            scoring = score_property(purchase_price, prediction['predicted_annual_revenue'],
                                      prediction['predicted_occupancy_rate'])
            target['predicted'] = prediction
            target['score'] = scoring['score']
            target['score_breakdown'] = scoring['breakdown']
            target['financials'] = scoring['financials']
        except Exception:
            pass

    target.setdefault('source', 'airbnb')
    target.setdefault('address', '')
    target.setdefault('photo_url', None)

    # Find comparable listings
    comps = get_comparable_listings(
        target['latitude'], target['longitude'],
        target.get('bedrooms'), radius_km=3, limit=10
    )

    # Neighborhood stats
    neighborhoods = get_neighborhood_stats()
    target_hood = target.get('neighbourhood', '')
    hood_stats = next((n for n in neighborhoods
                       if n['neighbourhood_cleansed'] == target_hood), None)

    return jsonify({
        'property': target,
        'comparables': comps,
        'neighborhood_stats': hood_stats,
    })
