import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

RENTCAST_API_KEY = os.getenv('RENTCAST_API_KEY', '')
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', '')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'trained')

_BACKEND_DIR = os.path.dirname(__file__)


def city_data_dir(city_key):
    """Return the data directory for a city, e.g. backend/data/cities/austin/"""
    d = os.path.join(_BACKEND_DIR, 'data', 'cities', city_key)
    os.makedirs(d, exist_ok=True)
    return d


def city_models_dir(city_key):
    """Return the models directory for a city, e.g. backend/models/trained/austin/"""
    d = os.path.join(_BACKEND_DIR, 'models', 'trained', city_key)
    os.makedirs(d, exist_ok=True)
    return d


STREETVIEW_CACHE_DIR = os.path.join(_BACKEND_DIR, 'data', 'streetview_cache')


def streetview_cache_dir():
    """Return the streetview cache directory, creating it if needed."""
    os.makedirs(STREETVIEW_CACHE_DIR, exist_ok=True)
    return STREETVIEW_CACHE_DIR


os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
