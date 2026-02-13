from flask import Flask
from flask_cors import CORS
import config

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Ensure data directories exist
    config.DATA_RAW_DIR
    config.DATA_PROCESSED_DIR
    config.TRAINED_MODELS_DIR

    from routes.properties import properties_bp
    from routes.analytics import analytics_bp
    from routes.predict import predict_bp
    from routes.settings import settings_bp

    app.register_blueprint(properties_bp, url_prefix='/api')
    app.register_blueprint(analytics_bp, url_prefix='/api')
    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(settings_bp, url_prefix='/api')

    @app.route('/api/health')
    def health():
        return {'status': 'ok'}

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
