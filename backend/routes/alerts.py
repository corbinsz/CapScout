"""Weekly deal-alert requests.

This stores the request. It does not send email. Delivery needs an email
provider the owner connects later (see docs/DEPLOY.md).
"""
from flask import Blueprint, jsonify, request

from security import bounded_float, city_or_400, rate_limit, valid_email

alerts_bp = Blueprint('alerts', __name__)


@alerts_bp.route('/alerts', methods=['POST'])
@rate_limit('alerts', limit=5, window=600)
def create_alert():
    from services.account_service import save_alert

    body = request.get_json(silent=True) or {}
    email = valid_email(body.get('email'))
    city_key, error = city_or_400(body.get('city'))
    if error:
        return error
    if not email:
        return jsonify({'error': 'Enter a valid email address.'}), 400
    try:
        min_score = bounded_float(body.get('min_score', 70), 0, 100)
    except ValueError:
        return jsonify({'error': 'Score threshold must be between 0 and 100.'}), 400
    if min_score is None:
        min_score = 70
    cadence = body.get('cadence') or 'weekly'
    if cadence != 'weekly':
        return jsonify({'error': 'Only a weekly alert is available right now.'}), 400
    saved = save_alert(email, city_key, min_score, cadence)
    return jsonify({
        'ok': True,
        'alert': saved,
        'delivery': 'stored',
        'message': 'Saved. Weekly email starts after an email provider is connected. Nothing was sent.',
    }), 201
