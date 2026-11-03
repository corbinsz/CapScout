"""
Settings API — save/read API keys, city list.
Keys are persisted to backend/.env and reloaded into config.
"""
import os
from flask import Blueprint, request, jsonify
import config
from cities import list_cities, CITIES
from services.rentcast_service import has_api_key as rentcast_has_key
from services.realtor_service import has_api_key as rapidapi_has_key
from services.redfin_service import has_data as redfin_has_data, save_csv as redfin_save_csv, row_count as redfin_row_count
from services.prediction_service import models_available
from services.providers.registry import provider_status
from security import city_or_400, rate_limit

settings_bp = Blueprint('settings', __name__)

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')


def _read_env():
    """Read existing .env file into a dict."""
    env = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    env[key.strip()] = value.strip().strip("'\"")
    return env


def _write_env(env):
    """Write dict back to .env file."""
    with open(ENV_PATH, 'w') as f:
        for key, value in env.items():
            f.write(f"{key}={value}\n")


@settings_bp.route('/settings/keys', methods=['GET'])
def get_keys_status():
    """Return which providers are configured. Never returns the key itself."""
    city_key, error = city_or_400(request.args.get('city'))
    if error:
        return error
    status = provider_status()
    return jsonify({
        'rentcast': rentcast_has_key(),
        'rapidapi': rapidapi_has_key(),
        'redfin_fallback_enabled': config.ALLOW_REDFIN_FALLBACK,
        'redfin': config.ALLOW_REDFIN_FALLBACK and redfin_has_data(city_key),
        'redfin_count': redfin_row_count(city_key) if redfin_has_data(city_key) else 0,
        'key_write_enabled': config.CAPSCOUT_ALLOW_KEY_WRITE,
        'listing_source': status['source'],
        'listing_label': status['label'],
        'listing_licensed': status['licensed'],
    })


@settings_bp.route('/settings/keys', methods=['POST'])
@rate_limit('key-write', limit=5, window=600)
def save_keys():
    """Dev-only. Production keys belong in the host's environment, not a POST body."""
    if not config.CAPSCOUT_ALLOW_KEY_WRITE:
        return jsonify({
            'error': 'API keys are read from server environment variables and cannot be set from the browser.',
        }), 403
    body = request.get_json(silent=True) or {}
    env = _read_env()
    if body.get('rentcast_key'):
        env['RENTCAST_API_KEY'] = str(body['rentcast_key'])[:200]
    if body.get('rapidapi_key'):
        env['RAPIDAPI_KEY'] = str(body['rapidapi_key'])[:200]
    _write_env(env)
    config.RENTCAST_API_KEY = env.get('RENTCAST_API_KEY', '')
    config.RAPIDAPI_KEY = env.get('RAPIDAPI_KEY', '')
    return jsonify({
        'ok': True,
        'rentcast': rentcast_has_key(),
        'rapidapi': rapidapi_has_key(),
    })


@settings_bp.route('/settings/redfin-upload', methods=['POST'])
@rate_limit('redfin-upload', limit=5, window=600)
def upload_redfin_csv():
    """Accept a Redfin CSV only when the dev fallback is explicitly enabled."""
    if not config.ALLOW_REDFIN_FALLBACK:
        return jsonify({
            'error': 'Redfin uploads are disabled. Set ALLOW_REDFIN_FALLBACK=1 for local development only.',
        }), 403
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename or not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Please upload a .csv file'}), 400

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > 20 * 1024 * 1024:
        return jsonify({'error': 'File is larger than 20 MB.'}), 413

    city_key, error = city_or_400(request.form.get('city'))
    if error:
        return error

    try:
        count = redfin_save_csv(file, city_key=city_key)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Invalidate scored-batch disk cache since listings changed
    from routes.properties import invalidate_disk_cache
    invalidate_disk_cache(city_key)

    return jsonify({'ok': True, 'count': count})


@settings_bp.route('/cities')
def get_cities():
    """GET /api/cities — list all supported cities with data/model status."""
    result = []
    for city_info in list_cities():
        key = city_info['key']
        city_dir = config.city_data_dir(key)
        has_data = (os.path.isfile(os.path.join(city_dir, 'listings.csv')) or
                    os.path.isfile(os.path.join(city_dir, 'listings.csv.gz')))
        has_models = models_available(city_key=key)
        result.append({
            'key': key,
            'name': city_info['name'],
            'state': city_info['state'],
            'display': city_info['display'],
            'latitude': city_info['latitude'],
            'longitude': city_info['longitude'],
            'default_zoom': city_info['default_zoom'],
            'has_data': has_data,
            'has_models': has_models,
        })
    return jsonify(result)
