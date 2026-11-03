from flask import Blueprint, jsonify, request
from services.data_service import get_market_overview
from services.providers.registry import provider_status
from security import city_or_400, rate_limit

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/analytics')
@rate_limit('analytics', limit=60, window=60)
def analytics():
    """GET /api/analytics — market overview, neighborhood rankings, distributions."""
    city_key, error = city_or_400(request.args.get('city'))
    if error:
        return error
    overview = get_market_overview(city_key=city_key)
    status = provider_status()
    overview['rentcast_connected'] = status['source'] == 'rentcast'
    overview['listing_source'] = status['source']
    overview['listing_licensed'] = status['licensed']
    return jsonify(overview)
