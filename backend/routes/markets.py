"""Cross-city ranking. The score formula is the same in every city, so a 70
in Austin can be compared with a 70 in Nashville.
"""
import time

from flask import Blueprint, jsonify, request

from cities import CITIES, get_city
from routes.properties import score_city_listings
from security import city_or_400, rate_limit

markets_bp = Blueprint('markets', __name__)

_cache = {'saved_at': 0, 'key': None, 'payload': None}
_TTL = 6 * 3600


@markets_bp.route('/markets/leaders')
@rate_limit('markets', limit=20, window=60)
def leaders():
    raw = request.args.get('cities', '')
    if raw.strip():
        keys = []
        for piece in raw.split(','):
            key, error = city_or_400(piece.strip())
            if error:
                return error
            keys.append(key)
    else:
        keys = list(CITIES.keys())
    keys = keys[:14]
    cache_key = tuple(keys)
    now = time.time()
    if _cache['payload'] is not None and _cache['key'] == cache_key and now - _cache['saved_at'] < _TTL:
        return jsonify(_cache['payload'])

    rows = []
    for key in keys:
        listings, status = score_city_listings(key, min_price=150000, max_price=750000)
        scored = [item for item in listings if item.get('score') is not None]
        if not scored:
            rows.append({
                'city': key,
                'display': get_city(key)['display'],
                'top_score': None,
                'message': status.get('message'),
            })
            continue
        best = max(scored, key=lambda item: item['score'])
        financials = best.get('financials') or {}
        rows.append({
            'city': key,
            'display': get_city(key)['display'],
            'top_score': best.get('score'),
            'address': best.get('address'),
            'property_id': best.get('id'),
            'price': best.get('price'),
            'monthly_cashflow': financials.get('monthly_cashflow'),
            'insight': best.get('insight'),
            'explanation': best.get('explanation'),
            'licensed': status.get('licensed'),
            'source': status.get('source'),
            'homes_scored': len(scored),
        })
    rows.sort(key=lambda row: row['top_score'] if row['top_score'] is not None else -1, reverse=True)
    payload = {
        'items': rows,
        'note': (
            'Each row is the highest score between $150k and $750k in that city, '
            'using the same loan assumptions. It is a screen, not a ranking of whole markets.'
        ),
    }
    _cache.update(saved_at=now, key=cache_key, payload=payload)
    return jsonify(payload)
