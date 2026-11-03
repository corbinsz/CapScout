import os
import secrets
import sys
import threading
from datetime import timedelta

from flask import Flask, request
from flask_cors import CORS
import config


def create_app():
    app = Flask(__name__)
    # A missing SECRET_KEY gets a process-local random value so sessions work
    # in dev without committing a secret. Set SECRET_KEY in production so
    # logins survive a restart.
    app.config['SECRET_KEY'] = config.SECRET_KEY or secrets.token_hex(32)
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=14)
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_SECURE'] = config._flag('SESSION_COOKIE_SECURE')
    CORS(app, origins=config.CORS_ORIGINS, supports_credentials=True)

    config.TRAINED_MODELS_DIR

    from routes.properties import properties_bp
    from routes.analytics import analytics_bp
    from routes.predict import predict_bp
    from routes.settings import settings_bp
    from routes.streetview import streetview_bp
    from routes.auth import auth_bp
    from routes.billing import billing_bp
    from routes.reports import reports_bp
    from routes.model_metrics import metrics_bp
    from routes.alerts import alerts_bp
    from routes.markets import markets_bp
    from routes.telemetry import telemetry_bp

    app.register_blueprint(properties_bp, url_prefix='/api')
    app.register_blueprint(analytics_bp, url_prefix='/api')
    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(settings_bp, url_prefix='/api')
    app.register_blueprint(streetview_bp, url_prefix='/api')
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(billing_bp, url_prefix='/api')
    app.register_blueprint(reports_bp, url_prefix='/api')
    app.register_blueprint(metrics_bp, url_prefix='/api')
    app.register_blueprint(alerts_bp, url_prefix='/api')
    app.register_blueprint(markets_bp, url_prefix='/api')
    app.register_blueprint(telemetry_bp, url_prefix='/api')

    @app.route('/api/health')
    def health():
        from services.providers.registry import provider_status
        return {'status': 'ok', 'listing_source': provider_status()['source']}

    @app.after_request
    def security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        if request.path.startswith('/api/auth') or request.path.startswith('/api/billing'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    return app


def _warmup():
    """Preload default city models so first request is fast.

    Only loads the default city to keep memory low on Render free tier (512MB).
    Other cities lazy-load on first request (~400ms one-time).
    """
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


print("[app] creating flask app ...", flush=True)
app = create_app()
print("[app] flask app created", flush=True)

# Skip warmup on Render to keep memory low — models lazy-load on first request
if not os.environ.get('RENDER') and not os.environ.get('SKIP_WARMUP'):
    threading.Thread(target=_warmup, daemon=True).start()
else:
    print("[app] Render detected, skipping warmup (lazy-load on demand)", flush=True)

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', '').strip() in ('1', 'true', 'yes'), port=5000)
