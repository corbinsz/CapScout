from flask import Blueprint, request, jsonify
from services.prediction_service import predict_property, models_available
from services.scoring_service import score_property

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Score a custom property by features.

    Body JSON:
    {
        "purchase_price": 450000,
        "bedrooms": 3,
        "bathrooms": 2,
        "accommodates": 6,
        "latitude": 30.27,
        "longitude": -97.74,
        "amenity_count": 15
    }
    """
    if not models_available():
        return jsonify({'error': 'Models not trained. Run train_model.py first.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    purchase_price = data.get('purchase_price', 0)
    if not purchase_price or purchase_price <= 0:
        return jsonify({'error': 'purchase_price is required and must be > 0'}), 400

    features = {
        'bedrooms': data.get('bedrooms', 2),
        'bathrooms_clean': data.get('bathrooms', 1),
        'accommodates': data.get('accommodates', data.get('bedrooms', 2) * 2),
        'latitude': data.get('latitude', 30.27),
        'longitude': data.get('longitude', -97.74),
    }

    if 'amenity_count' in data:
        features['amenity_count'] = data['amenity_count']

    try:
        prediction = predict_property(features)
        scoring = score_property(
            purchase_price,
            prediction['predicted_annual_revenue'],
            prediction['predicted_occupancy_rate']
        )

        return jsonify({
            'prediction': prediction,
            'score': scoring['score'],
            'score_breakdown': scoring['breakdown'],
            'financials': scoring['financials'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
