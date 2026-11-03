from flask import Blueprint, jsonify, request
from cities import get_city_or_default
from services.prediction_service import predict_property, models_available
from services.scoring_service import explain_deal, score_property
from security import bounded_float, bounded_int, city_or_400, rate_limit

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
@rate_limit('predict', limit=30, window=60)
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
        "amenity_count": 15,
        "city": "austin"
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    city_key, error = city_or_400(data.get('city'))
    if error:
        return error
    city = get_city_or_default(city_key)

    if not models_available(city_key):
        return jsonify({'error': f'Models not trained for {city["display"]}. Run train_model.py first.'}), 503

    try:
        purchase_price = bounded_float(data.get('purchase_price'), 1, 100_000_000)
        bedrooms = bounded_int(data.get('bedrooms', 2), 0, 20) or 0
        bathrooms = bounded_float(data.get('bathrooms', 1), 0, 20)
        latitude = bounded_float(data.get('latitude', city['latitude']), -90, 90)
        longitude = bounded_float(data.get('longitude', city['longitude']), -180, 180)
    except ValueError:
        return jsonify({'error': 'One of the property values is missing or out of range.'}), 400
    if not purchase_price:
        return jsonify({'error': 'purchase_price is required and must be > 0'}), 400

    features = {
        'bedrooms': bedrooms,
        'bathrooms_clean': bathrooms if bathrooms is not None else 1,
        'accommodates': data.get('accommodates') or bedrooms * 2,
        'latitude': latitude if latitude is not None else city['latitude'],
        'longitude': longitude if longitude is not None else city['longitude'],
    }

    if 'amenity_count' in data:
        features['amenity_count'] = data['amenity_count']

    # Optional month for seasonal occupancy prediction (1-12)
    month = data.get('month')
    if month is not None:
        month = int(month)
        if month < 1 or month > 12:
            return jsonify({'error': 'month must be 1-12'}), 400

    try:
        prediction = predict_property(features, month=month, city_key=city_key)
        scoring = score_property(
            purchase_price,
            prediction['predicted_annual_revenue'],
            prediction['predicted_occupancy_rate'],
            city_key=city_key,
        )

        response = {
            'prediction': prediction,
            'score': scoring['score'],
            'score_breakdown': scoring['breakdown'],
            'financials': scoring['financials'],
            'explanation': explain_deal(
                scoring['score'],
                scoring['financials'],
                (prediction.get('confidence') or {}).get('level'),
            ),
        }
        if 'monthly_breakdown' in prediction:
            response['monthly_breakdown'] = prediction['monthly_breakdown']
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
