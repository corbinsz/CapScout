from flask import Blueprint, jsonify
from services.data_service import get_market_overview, get_neighborhood_stats
from services.rentcast_service import has_api_key

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/analytics')
def analytics():
    """GET /api/analytics — market overview, neighborhood rankings, distributions."""
    overview = get_market_overview()
    overview['rentcast_connected'] = has_api_key()
    return jsonify(overview)
