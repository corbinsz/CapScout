"""
Settings API — save/read API keys.
Keys are persisted to backend/.env and reloaded into config.
"""
import os
from flask import Blueprint, request, jsonify
import config
from services.rentcast_service import has_api_key as rentcast_has_key
from services.realtor_service import has_api_key as rapidapi_has_key
from services.redfin_service import has_data as redfin_has_data, save_csv as redfin_save_csv, row_count as redfin_row_count

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
    """Return which API keys are configured (not the keys themselves)."""
    return jsonify({
        'rentcast': rentcast_has_key(),
        'rapidapi': rapidapi_has_key(),
        'redfin': redfin_has_data(),
        'redfin_count': redfin_row_count() if redfin_has_data() else 0,
    })


@settings_bp.route('/settings/keys', methods=['POST'])
def save_keys():
    """Save API keys to .env and reload config."""
    body = request.get_json(force=True)
    env = _read_env()

    if 'rentcast_key' in body and body['rentcast_key']:
        env['RENTCAST_API_KEY'] = body['rentcast_key']
    if 'rapidapi_key' in body and body['rapidapi_key']:
        env['RAPIDAPI_KEY'] = body['rapidapi_key']

    _write_env(env)

    # Reload config values
    config.RENTCAST_API_KEY = env.get('RENTCAST_API_KEY', '')
    config.RAPIDAPI_KEY = env.get('RAPIDAPI_KEY', '')

    return jsonify({
        'ok': True,
        'rentcast': bool(config.RENTCAST_API_KEY) and config.RENTCAST_API_KEY != 'your_api_key_here',
        'rapidapi': bool(config.RAPIDAPI_KEY) and config.RAPIDAPI_KEY != 'your_api_key_here',
    })


@settings_bp.route('/settings/redfin-upload', methods=['POST'])
def upload_redfin_csv():
    """Accept a Redfin CSV upload (multipart/form-data)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename or not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Please upload a .csv file'}), 400

    try:
        count = redfin_save_csv(file)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'ok': True, 'count': count})
