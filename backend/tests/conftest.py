import os
import tempfile

_db = tempfile.NamedTemporaryFile(prefix='capscout-test-', suffix='.sqlite', delete=False)
os.environ['ACCOUNTS_DB'] = _db.name
os.environ['SKIP_WARMUP'] = '1'
os.environ['SECRET_KEY'] = 'test-secret-not-for-production'
os.environ['ALLOW_REDFIN_FALLBACK'] = '1'
os.environ['CAPSCOUT_ALLOW_DEV_BILLING'] = '1'
os.environ['ANALYTICS_ENABLED'] = '1'
os.environ['CLIENT_ERROR_LOG'] = '1'

import pytest
from security import reset_rate_limits


@pytest.fixture
def client():
    reset_rate_limits()
    from app import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app.test_client()
