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
    """Preload default city models so first request is fast.

    Only loads the default city to keep memory low on Render free tier (512MB).
    Other cities lazy-load on first request (~400ms one-time).
    """
    import sys
    try:
        from cities import DEFAULT_CITY
        from services.prediction_service import _load_models, _get_balltree
        from services.redfin_service import get_listings

        _load_models(DEFAULT_CITY)
        _get_balltree(DEFAULT_CITY)
        get_listings(DEFAULT_CITY)
        print(f"[warmup] {DEFAULT_CITY} ready", flush=True)
    except Exception as e:
        print(f"[warmup] failed: {e}", file=sys.stderr, flush=True)


app = create_app()

# Warm up caches in background thread so Render health checks pass immediately
threading.Thread(target=_warmup, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
