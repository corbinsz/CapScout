import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

RENTCAST_API_KEY = os.getenv('RENTCAST_API_KEY', '')
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', '')
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'trained')

# Ensure directories exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, TRAINED_MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
