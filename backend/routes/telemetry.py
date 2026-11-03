"""Optional, privacy-light hooks.

Analytics and client-error logging are off unless the owner turns them on.
Events are counts by day. They are not tied to an account. Client errors drop
anything that looks like an email address.
"""
import os
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from security import clean_text, rate_limit

telemetry_bp = Blueprint('telemetry', __name__)

_EVENTS = {'page_view', 'city_selected', 'report_clicked', 'alert_saved', 'deal_shared'}
_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'client_errors.log')


def _flag(name):
    return os.getenv(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


@telemetry_bp.route('/events', methods=['POST'])
@rate_limit('events', limit=60, window=60)
def events():
    if not _flag('ANALYTICS_ENABLED'):
        return jsonify({'stored': False})
    body = request.get_json(silent=True) or {}
    name = clean_text(body.get('name'), 40)
    if name not in _EVENTS:
        return jsonify({'error': 'Unknown event.'}), 400
    from services.account_service import record_event
    record_event(name)
    return jsonify({'stored': True})


@telemetry_bp.route('/client-errors', methods=['POST'])
@rate_limit('client-errors', limit=20, window=60)
def client_errors():
    if not _flag('CLIENT_ERROR_LOG'):
        return ('', 204)
    body = request.get_json(silent=True) or {}
    message = clean_text(body.get('message'), 240)
    path = clean_text(body.get('path'), 120)
    if '@' in message or '@' in path:
        return ('', 204)
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(_LOG_PATH, 'a', encoding='utf-8') as handle:
        handle.write(f'{stamp}\t{path}\t{message}\n')
    return ('', 204)
