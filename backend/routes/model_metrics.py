"""Serve the holdout metrics written next to each city's model."""
import json
import os

from flask import Blueprint, jsonify, request

import config
from security import city_or_400

metrics_bp = Blueprint('model_metrics', __name__)


@metrics_bp.route('/model-metrics')
def model_metrics():
    city_key, error = city_or_400(request.args.get('city'))
    if error:
        return error
    path = os.path.join(config.city_models_dir(city_key), 'metrics.json')
    if not os.path.isfile(path):
        return jsonify({
            'available': False,
            'city': city_key,
            'note': 'Holdout metrics have not been saved for this city yet.',
        })
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return jsonify({'available': False, 'city': city_key}), 500
    payload['available'] = True
    payload['city'] = city_key
    return jsonify(payload)