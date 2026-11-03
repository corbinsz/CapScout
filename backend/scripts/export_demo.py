"""Write the Austin preview bundle the website can open with no API server.

From backend/:
    ALLOW_REDFIN_FALLBACK=1 python -m scripts.export_demo
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('ALLOW_REDFIN_FALLBACK', '1')
os.environ.setdefault('SKIP_WARMUP', '1')
os.environ.setdefault('SECRET_KEY', 'demo')

from app import create_app

KEEP = {
    'id', 'address', 'name', 'price', 'bedrooms', 'bathrooms', 'sqft', 'year_built',
    'lot_size', 'latitude', 'longitude', 'property_type', 'hoa_monthly', 'days_on_market',
    'price_per_sqft', 'score', 'confidence_level', 'explanation', 'insight', 'source',
    'url', 'score_breakdown', 'financials', 'predicted',
}


def slim(item):
    out = {key: item.get(key) for key in KEEP}
    predicted = out.get('predicted') or {}
    out['predicted'] = {
        'predicted_nightly_rate': predicted.get('predicted_nightly_rate'),
        'predicted_occupancy_rate': predicted.get('predicted_occupancy_rate'),
        'predicted_monthly_revenue': predicted.get('predicted_monthly_revenue'),
        'predicted_annual_revenue': predicted.get('predicted_annual_revenue'),
        'confidence': predicted.get('confidence'),
    }
    out['photo_url'] = None
    return out


def main():
    client = create_app().test_client()
    items = []
    page = 1
    while page < 80:
        response = client.get(
            '/api/properties',
            query_string={
                'city': 'austin',
                'page': page,
                'per_page': 50,
                'min_price': 100000,
                'max_price': 2000000,
                'str_viable': 'false',
                'sort_by': 'score',
                'sort_order': 'desc',
            },
        )
        body = response.get_json()
        batch = body.get('items') or []
        items.extend(slim(item) for item in batch)
        if page * 50 >= (body.get('total') or 0) or not batch:
            message = body.get('message')
            break
        page += 1
    else:
        message = ''

    metrics = client.get('/api/model-metrics', query_string={'city': 'austin'}).get_json()
    # Drop bulky nested training dumps if any.
    if isinstance(metrics, dict):
        metrics.pop('before', None)

    payload = {
        'city': 'austin',
        'display': 'Austin, TX',
        'message': (
            'This preview is a saved Austin sample, so it opens without a server. '
            'Live listings for every city turn on when a RentCast key is connected.'
        ),
        'metrics': metrics,
        'assumptions': {
            'taxRate': 0.018,
            'insuranceRate': 0.007,
            'strPermitAnnual': 0,
            'cleaningPerTurnover': 125,
            'avgStayNights': 3.2,
            'baseUtilitiesMonthly': 275,
        },
        'listings': items,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'public', 'demo')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'austin.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, separators=(',', ':'))
    print(f'wrote {len(items)} listings to {path} ({os.path.getsize(path) / 1e6:.1f} MB)')
    print('note:', (message or '')[:140])


if __name__ == '__main__':
    main()
