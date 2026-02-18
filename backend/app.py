import logging
from flask import Flask
from flask_cors import CORS
import config

log = logging.getLogger(__name__)


def _warmup():
    """Pre-load ML models and Redfin data so the first request is fast."""
    from cities import CITIES
    from services.prediction_service import _load_models
    from services.redfin_service import get_listings, has_data

    for city_key in CITIES:
        try:
            _load_models(city_key)
        except Exception:
            pass
        if has_data(city_key):
            try:
                get_listings(city_key)
            except Exception:
                pass
    log.info('Warmup complete: models and Redfin data cached')


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Ensure base directories exist
    config.TRAINED_MODELS_DIR

    from routes.properties import properties_bp
    from routes.analytics import analytics_bp
    from routes.predict import predict_bp
    from routes.settings import settings_bp
    from routes.streetview import streetview_bp

    app.register_blueprint(properties_bp, url_prefix='/api')
    app.register_blueprint(analytics_bp, url_prefix='/api')
    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(settings_bp, url_prefix='/api')
    app.register_blueprint(streetview_bp, url_prefix='/api')

    @app.route('/api/health')
    def health():
        return {'status': 'ok'}

    _warmup()

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
