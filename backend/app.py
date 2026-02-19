import threading
from flask import Flask
from flask_cors import CORS
import config


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

    return app


def _warmup():
    """Preload models, BallTrees, and Redfin CSVs so first request is fast."""
    from cities import CITIES
    from services.prediction_service import _load_models, _get_balltree
    from services.redfin_service import get_listings

    for city_key in CITIES:
        _load_models(city_key)
        _get_balltree(city_key)
        get_listings(city_key)


app = create_app()

# Warm up caches in background thread so Render health checks pass immediately
threading.Thread(target=_warmup, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
