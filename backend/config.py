import os
import re

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

_BACKEND_DIR = os.path.dirname(__file__)


def _flag(name, default='0'):
    return os.getenv(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


def _clean_secret(value):
    """Treat blank and placeholder values as unset. Never log the raw value."""
    value = (value or '').strip()
    if not value or value.lower() in {
        'your_api_key_here', 'changeme', 'placeholder', 'sk_test_placeholder', 'sk_live_placeholder',
    }:
        return ''
    return value


RENTCAST_API_KEY = _clean_secret(os.getenv('RENTCAST_API_KEY', ''))
RAPIDAPI_KEY = _clean_secret(os.getenv('RAPIDAPI_KEY', ''))
GOOGLE_MAPS_API_KEY = _clean_secret(os.getenv('GOOGLE_MAPS_API_KEY', ''))
STRIPE_SECRET_KEY = _clean_secret(os.getenv('STRIPE_SECRET_KEY', ''))
STRIPE_PRICE_ID = os.getenv('STRIPE_PRICE_ID', '').strip()
STRIPE_WEBHOOK_SECRET = _clean_secret(os.getenv('STRIPE_WEBHOOK_SECRET', ''))
SECRET_KEY = os.getenv('SECRET_KEY', '').strip()

# Redfin's map endpoint is not licensed for a commercial product.
# The saved CSV can be read only when this is explicitly turned on for local dev.
ALLOW_REDFIN_FALLBACK = _flag('ALLOW_REDFIN_FALLBACK')
ALLOW_REDFIN_DOWNLOAD = _flag('ALLOW_REDFIN_DOWNLOAD')
CAPSCOUT_ALLOW_KEY_WRITE = _flag('CAPSCOUT_ALLOW_KEY_WRITE')
CAPSCOUT_ALLOW_DEV_BILLING = _flag('CAPSCOUT_ALLOW_DEV_BILLING')

RENTCAST_MAX_PAGES = max(1, min(int(os.getenv('RENTCAST_MAX_PAGES', '1') or 1), 10))
RENTCAST_CACHE_HOURS = max(1, min(int(os.getenv('RENTCAST_CACHE_HOURS', '24') or 24), 24 * 14))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        'CORS_ORIGINS',
        'http://localhost:3000,http://127.0.0.1:3000',
    ).split(',')
    if origin.strip()
]

TRAINED_MODELS_DIR = os.path.join(_BACKEND_DIR, 'models', 'trained')
DATA_CACHE_DIR = os.path.join(_BACKEND_DIR, 'data', 'cache')
# Older clients imported this name. It did not exist, so RentCast caching crashed.
DATA_PROCESSED_DIR = DATA_CACHE_DIR
ACCOUNTS_DB_PATH = os.getenv('ACCOUNTS_DB') or os.path.join(_BACKEND_DIR, 'data', 'accounts.sqlite')
STREETVIEW_CACHE_DIR = os.path.join(_BACKEND_DIR, 'data', 'streetview_cache')

_CITY_KEY = re.compile(r'^[a-z0-9-]{1,40}$')


def is_known_city(city_key):
    """True when city_key is a registered market. Rejects path traversal."""
    if not city_key or not _CITY_KEY.match(str(city_key)):
        return False
    from cities import CITIES
    return city_key in CITIES


def city_data_dir(city_key):
    """Return the data directory for a registered city.

    Unknown keys raise KeyError instead of creating directories from user input.
    """
    if not is_known_city(city_key):
        raise KeyError(f'Unknown city: {city_key!r}')
    d = os.path.join(_BACKEND_DIR, 'data', 'cities', city_key)
    os.makedirs(d, exist_ok=True)
    return d


def city_models_dir(city_key):
    """Return the models directory for a registered city."""
    if not is_known_city(city_key):
        raise KeyError(f'Unknown city: {city_key!r}')
    d = os.path.join(_BACKEND_DIR, 'models', 'trained', city_key)
    os.makedirs(d, exist_ok=True)
    return d


def streetview_cache_dir():
    os.makedirs(STREETVIEW_CACHE_DIR, exist_ok=True)
    return STREETVIEW_CACHE_DIR


def cache_dir():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    return DATA_CACHE_DIR


os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ACCOUNTS_DB_PATH), exist_ok=True)
