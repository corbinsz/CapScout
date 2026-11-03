def test_health(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json['status'] == 'ok'


def test_unknown_city_is_400(client):
    response = client.get('/api/properties?city=../../.env')
    assert response.status_code == 400


def test_keys_cannot_be_posted(client):
    response = client.post('/api/settings/keys', json={'rentcast_key': 'secret-value'})
    assert response.status_code == 403
    assert 'secret-value' not in response.get_data(as_text=True)


def test_register_login_and_paywall(client):
    created = client.post('/api/auth/register', json={
        'email': 'buyer@example.com',
        'password': 'correct horse',
    })
    assert created.status_code == 201
    assert 'password' not in created.json['user']

    bad = client.post('/api/auth/login', json={
        'email': 'buyer@example.com',
        'password': 'wrong-password',
    })
    assert bad.status_code == 401
    assert bad.json['error'] == 'Email or password is incorrect.'

    report = client.get('/api/reports/top-deals?city=asheville')
    assert report.status_code == 402

    upgraded = client.post('/api/billing/dev-activate')
    assert upgraded.status_code == 200
    report = client.get('/api/reports/top-deals?city=asheville&limit=5')
    assert report.status_code == 200
    assert 'not financial' in report.get_data(as_text=True).lower()
    assert 'asheville' in report.headers['Content-Disposition']


def test_alert_stores_without_sending(client):
    response = client.post('/api/alerts', json={
        'email': 'owner@example.com',
        'city': 'austin',
        'min_score': 70,
        'cadence': 'weekly',
    })
    assert response.status_code == 201
    assert response.json['delivery'] == 'stored'
    assert 'Nothing was sent' in response.json['message']


def test_honeypot_rejected(client):
    response = client.post('/api/auth/register', json={
        'email': 'bot@example.com',
        'password': 'longenough1',
        'company': 'spam llc',
    })
    assert response.status_code == 400


def test_rate_limit_on_login(client):
    for _ in range(8):
        client.post('/api/auth/login', json={'email': 'a@b.co', 'password': 'wrong-password-1'})
    blocked = client.post('/api/auth/login', json={'email': 'a@b.co', 'password': 'wrong-password-1'})
    assert blocked.status_code == 429


def test_deals_endpoint_returns_scores(client):
    response = client.get('/api/properties?city=asheville&per_page=2&page=1')
    assert response.status_code == 200
    body = response.json
    assert body['data_source'] == 'redfin'
    assert body['data_licensed'] is False
    assert body['items']
    assert body['items'][0]['explanation']
    assert 'score' in body['items'][0]


def test_checkout_is_a_stub_without_stripe(client):
    client.post('/api/auth/register', json={
        'email': 'pay@example.com',
        'password': 'longenough1',
    })
    response = client.post('/api/billing/checkout')
    assert response.status_code == 200
    assert response.json['mode'] == 'stub'
    assert response.json['configured'] is False


def test_event_allowlist(client):
    ok = client.post('/api/events', json={'name': 'page_view'})
    assert ok.status_code == 200
    assert ok.json['stored'] is True
    bad = client.post('/api/events', json={'name': 'steal_email'})
    assert bad.status_code == 400
