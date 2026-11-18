import os
import sys
import threading
from pathlib import Path

from flask import Flask
from flask_cors import CORS

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(1, str(_ROOT))

import config


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Ensure base directories exist
    config.TRAINED_MODELS_DIR

    from routes.analytics import analytics_bp
    from routes.predict import predict_bp
    from routes.properties import properties_bp
    from routes.settings import settings_bp
    from routes.streetview import streetview_bp

    app.register_blueprint(properties_bp, url_prefix="/api")
    app.register_blueprint(analytics_bp, url_prefix="/api")
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(settings_bp, url_prefix="/api")
    app.register_blueprint(streetview_bp, url_prefix="/api")

    @app.route("/api/health")
    def health():
        return {"status": "ok"}

    return app


def _warmup():
    """Preload the default city's model so the first local request is faster."""
    try:
        from cities import DEFAULT_CITY
        from services.prediction_service import _balltree, _load

        _load(DEFAULT_CITY)
        _balltree(DEFAULT_CITY)
        print(f"[warmup] {DEFAULT_CITY} ready", flush=True)
    except Exception as exc:
        print(f"[warmup] failed: {exc}", file=sys.stderr, flush=True)


print("[app] creating flask app ...", flush=True)
app = create_app()
print("[app] flask app created", flush=True)

# A small process can set SKIP_WARMUP=1 and load a model on the first request.
if not os.environ.get("SKIP_WARMUP"):
    threading.Thread(target=_warmup, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
