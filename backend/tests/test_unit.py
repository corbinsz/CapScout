import numpy as np
import joblib
from sklearn.linear_model import Ridge

from models.estimators import DollarRegressor
from models.metrics import group_holdout, regression_metrics
from security import parse_city, valid_email, valid_password
from services.providers.rentcast_provider import normalize_rentcast
from services.redfin_service import normalize_listing
from services.scoring_service import calculate_annual_mortgage, explain_deal, score_property


def test_mortgage_matches_standard_amortization():
    annual = calculate_annual_mortgage(200000, rate=0.06, years=30)
    assert abs(annual - 1199.10 * 12) < 2


def test_score_is_bounded_and_explained():
    result = score_property(400000, 48000, 0.62, city_key='austin')
    assert 0 <= result['score'] <= 100
    text = explain_deal(result['score'], result['financials'], 'high')
    assert 'not financial advice' in text.lower()


def test_zero_price_scores_zero():
    assert score_property(0, 1000, 0.5)['score'] == 0


def test_city_rejects_path_traversal():
    assert parse_city('../../etc') is None
    assert parse_city('austin') == 'austin'
    assert parse_city('Austin') == 'austin'


def test_email_and_password_rules():
    assert valid_email('Owner@Example.com') == 'owner@example.com'
    assert valid_email('not-an-email') is None
    assert valid_email('a@b\n.com') is None
    assert valid_password('short') is None
    assert valid_password('longenough1') == 'longenough1'


def test_rentcast_keeps_half_baths_and_drops_land():
    home = normalize_rentcast({
        'id': 'abc123',
        'formattedAddress': '1 Main St, Austin, TX 78701',
        'status': 'Active',
        'propertyType': 'Single Family',
        'price': 450000,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'squareFootage': 1600,
        'latitude': 30.27,
        'longitude': -97.74,
    })
    assert home['bathrooms'] == 2.5
    assert home['id'].startswith('rentcast-')
    assert normalize_rentcast({
        'formattedAddress': 'Lot 9',
        'status': 'Active',
        'propertyType': 'Land',
        'price': 100000,
        'bedrooms': 0,
    }) is None


def test_redfin_keeps_half_baths():
    row = normalize_listing({
        'ADDRESS': '9 Oak',
        'CITY': 'Austin',
        'STATE OR PROVINCE': 'TX',
        'ZIP OR POSTAL CODE': '78704',
        'PRICE': '300000',
        'BEDS': '3',
        'BATHS': '2.5',
        'LATITUDE': '30.25',
        'LONGITUDE': '-97.75',
        'PROPERTY TYPE': 'Single Family Residential',
    })
    assert row['bathrooms'] == 2.5
    assert row['source'] == 'redfin'


def test_holdout_groups_do_not_overlap():
    ids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    train, test = group_holdout(ids, test_size=0.3, seed=1)
    assert set(ids[train]).isdisjoint(set(ids[test]))


def test_regression_metrics_perfect():
    stats = regression_metrics([1, 2, 3], [1, 2, 3])
    assert stats['mae'] == 0
    assert stats['r2'] == 1


def test_dollar_model_survives_pickle(tmp_path):
    model = Ridge().fit(np.array([[0.0], [1.0], [2.0]]), np.log1p([100, 150, 220]))
    wrapped = DollarRegressor(model, 'price')
    path = tmp_path / 'model.joblib'
    joblib.dump(wrapped, path)
    loaded = joblib.load(path)
    pred = float(loaded.predict([[1.0]])[0])
    assert 50 < pred < 400


def test_saved_city_model_predicts_dollars():
    from services.prediction_service import predict_property
    result = predict_property(
        {
            'bedrooms': 3,
            'bathrooms_clean': 2,
            'accommodates': 6,
            'latitude': 30.27,
            'longitude': -97.74,
        },
        city_key='austin',
        summary_only=True,
    )
    assert 30 < result['predicted_nightly_rate'] < 2500
    assert 0 < result['predicted_occupancy_rate'] < 1
