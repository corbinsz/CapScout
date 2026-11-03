"""Paid 'Top deals in [city]' export. Free accounts get a 402 and no rows."""
import csv
import io

from flask import Blueprint, Response, jsonify, request

from cities import get_city
from routes.auth import current_user
from routes.properties import score_city_listings
from security import city_or_400, rate_limit

reports_bp = Blueprint('reports', __name__)


@reports_bp.route('/reports/top-deals')
@rate_limit('report', limit=10, window=600)
def top_deals():
    city_key, error = city_or_400(request.args.get('city'))
    if error:
        return error
    user = current_user()
    if not user:
        return jsonify({
            'error': 'Create a free account to download reports.',
            'needs_auth': True,
        }), 401
    if user['tier'] != 'pro':
        return jsonify({
            'error': 'The Top deals report is part of CapScout Pro.',
            'upgrade_required': True,
        }), 402

    try:
        limit = int(request.args.get('limit', 25))
    except ValueError:
        return jsonify({'error': 'limit must be a number.'}), 400
    limit = max(5, min(limit, 50))

    rows, status = score_city_listings(city_key)
    if status.get('message') and not rows:
        return jsonify({'error': status['message']}), 404

    ranked = sorted(rows, key=lambda item: item.get('score') or -1, reverse=True)[:limit]
    city = get_city(city_key)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        f"CapScout Top deals in {city['display']}",
    ])
    writer.writerow([
        'Estimates only. Not financial, investment, tax, or legal advice. '
        'Verify permits, insurance, taxes, and expenses before making an offer.',
    ])
    writer.writerow([
        'rank', 'address', 'price', 'bedrooms', 'bathrooms', 'score',
        'nightly_rate', 'occupancy', 'monthly_cashflow', 'cap_rate_pct',
        'dscr', 'confidence', 'explanation', 'source',
    ])
    for rank, item in enumerate(ranked, start=1):
        predicted = item.get('predicted') or {}
        financials = item.get('financials') or {}
        writer.writerow([
            rank,
            item.get('address') or item.get('name'),
            item.get('price'),
            item.get('bedrooms'),
            item.get('bathrooms'),
            item.get('score'),
            predicted.get('predicted_nightly_rate'),
            predicted.get('predicted_occupancy_rate'),
            financials.get('monthly_cashflow'),
            financials.get('cap_rate'),
            financials.get('dscr'),
            item.get('confidence_level'),
            item.get('explanation'),
            item.get('source'),
        ])
    filename = f"capscout-top-deals-{city_key}.csv"
    return Response(
        buffer.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )
