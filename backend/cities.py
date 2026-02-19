"""
City registry — single source of truth for all supported markets.
"""
import math

CITIES = {
    'austin': {
        'name': 'Austin',
        'state': 'TX',
        'display': 'Austin, TX',
        'latitude': 30.2672,
        'longitude': -97.7431,
        'default_zoom': 11,
        'property_tax_rate': 0.018,       # TX has no income tax but high property tax
        'insurance_rate': 0.007,          # TX highest in US — storms, hail, wind
        'str_permit_annual': 0,           # Austin Type 2 STR permit is one-time, no annual fee
        'cleaning_per_turnover': 125,     # mid-COL Texas
        'avg_stay_nights': 3.2,           # urban tourism + events
        'base_utilities_monthly': 275,    # hot summers, AC heavy
        'ia_state': 'tx',
        'ia_city': 'austin',
        'ia_date': '2025-06-13',
        'redfin_region_id': 30818,
        'assessor': {
            'source': 'travis_cad',
            'download_url': 'https://traviscad.org/publicinformation',
            'files': ['improvement_detail_2025.zip'],
        },
    },
    'nashville': {
        'name': 'Nashville',
        'state': 'TN',
        'display': 'Nashville, TN',
        'latitude': 36.1627,
        'longitude': -86.7816,
        'default_zoom': 11,
        'property_tax_rate': 0.0115,
        'insurance_rate': 0.005,          # TN moderate — some storm risk
        'str_permit_annual': 350,         # Nashville STR permit ~$313-350/yr
        'cleaning_per_turnover': 100,     # lower COL
        'avg_stay_nights': 3.0,           # weekend tourism
        'base_utilities_monthly': 225,    # moderate climate
        'ia_state': 'tn',
        'ia_city': 'nashville',
        'ia_date': '2025-06-19',  # calendar from 2024-06-22
        'redfin_region_id': 13415,
        'assessor': None,  # Davidson County: bulk building data not freely available
    },
    'san-diego': {
        'name': 'San Diego',
        'state': 'CA',
        'display': 'San Diego, CA',
        'latitude': 32.7157,
        'longitude': -117.1611,
        'default_zoom': 11,
        'property_tax_rate': 0.0076,      # CA Prop 13 caps at ~1% + local
        'insurance_rate': 0.004,          # CA moderate, fire zones vary
        'str_permit_annual': 1000,        # SD TOT registration + compliance costs
        'cleaning_per_turnover': 150,     # high COL California
        'avg_stay_nights': 4.0,           # beach vacation
        'base_utilities_monthly': 200,    # mild climate, minimal HVAC
        'ia_state': 'ca',
        'ia_city': 'san-diego',
        'ia_date': '2025-03-16',
        'redfin_region_id': 16904,
        'assessor': None,  # San Diego County: building data restricted
    },
    'dallas': {
        'name': 'Dallas',
        'state': 'TX',
        'display': 'Dallas, TX',
        'latitude': 32.7767,
        'longitude': -96.7970,
        'default_zoom': 11,
        'property_tax_rate': 0.018,       # TX high property tax
        'insurance_rate': 0.007,          # TX highest — storms, hail, tornado
        'str_permit_annual': 0,           # Dallas no STR permit required
        'cleaning_per_turnover': 100,     # mid-low COL
        'avg_stay_nights': 2.5,           # business travel market
        'base_utilities_monthly': 275,    # hot summers, AC heavy
        'ia_state': 'tx',
        'ia_city': 'dallas',
        'ia_date': '2025-11-23',  # upgraded: real calendar data now available
        'redfin_region_id': 30794,
        'assessor': {
            'source': 'dallas_cad',
            'download_url': 'https://www.dallascad.org/DataProducts.aspx',
            'files': ['DCAD2025_CERTIFIED_07242025.zip'],
        },
    },
    'asheville': {
        'name': 'Asheville',
        'state': 'NC',
        'display': 'Asheville, NC',
        'latitude': 35.5951,
        'longitude': -82.5515,
        'default_zoom': 12,
        'property_tax_rate': 0.0063,      # NC relatively low
        'insurance_rate': 0.004,          # NC moderate, inland = lower
        'str_permit_annual': 200,         # Buncombe County STR permit
        'cleaning_per_turnover': 90,      # low COL
        'avg_stay_nights': 3.5,           # mountain getaway
        'base_utilities_monthly': 200,    # moderate climate
        'ia_state': 'nc',
        'ia_city': 'asheville',
        'ia_date': '2023-12-18',
        'redfin_region_id': 555,
        'assessor': None,  # Buncombe County: building data limited
    },
    'seattle': {
        'name': 'Seattle',
        'state': 'WA',
        'display': 'Seattle, WA',
        'latitude': 47.6062,
        'longitude': -122.3321,
        'default_zoom': 11,
        'property_tax_rate': 0.009,
        'insurance_rate': 0.003,          # WA low risk — no hurricanes, minimal quake coverage
        'str_permit_annual': 75,          # Seattle STR operator license
        'cleaning_per_turnover': 160,     # high COL PNW
        'avg_stay_nights': 3.0,           # urban tourism
        'base_utilities_monthly': 225,    # mild summers, some heating
        'ia_state': 'wa',
        'ia_city': 'seattle',
        'ia_date': '2025-03-16',
        'redfin_region_id': 16163,
        'assessor': {
            'source': 'king_county',
            'download_url': 'https://info.kingcounty.gov/assessor/datadownload/default.aspx',
            'files': ['EXTR_ResBldg.csv', 'EXTR_Parcel.csv'],
            'arcgis_parcels': 'https://gismaps.kingcounty.gov/arcgis/rest/services/Property/KingCo_Parcels/MapServer/0',
        },
    },
    'denver': {
        'name': 'Denver',
        'state': 'CO',
        'display': 'Denver, CO',
        'latitude': 39.7392,
        'longitude': -104.9903,
        'default_zoom': 11,
        'property_tax_rate': 0.005,       # CO relatively low
        'insurance_rate': 0.004,          # CO moderate — hail risk
        'str_permit_annual': 200,         # Denver STR license ~$200/yr
        'cleaning_per_turnover': 130,     # mid-high COL
        'avg_stay_nights': 3.5,           # ski + outdoor
        'base_utilities_monthly': 250,    # cold winters, heating
        'ia_state': 'co',
        'ia_city': 'denver',
        'ia_date': '2025-09-29',
        'redfin_region_id': 5155,
        'assessor': None,
    },
    'new-orleans': {
        'name': 'New Orleans',
        'state': 'LA',
        'display': 'New Orleans, LA',
        'latitude': 29.9511,
        'longitude': -90.0715,
        'default_zoom': 12,
        'property_tax_rate': 0.011,       # LA moderate
        'insurance_rate': 0.008,          # LA high — hurricane risk
        'str_permit_annual': 500,         # NOLA strict STR regulation
        'cleaning_per_turnover': 110,     # lower COL
        'avg_stay_nights': 3.5,           # tourism / festivals
        'base_utilities_monthly': 275,    # hot, humid, AC heavy
        'ia_state': 'la',
        'ia_city': 'new-orleans',
        'ia_date': '2025-09-11',
        'redfin_region_id': 14233,
        'assessor': None,
    },
    # portland: removed — Dec 2025 IA scrape has zero price data
    'washington-dc': {
        'name': 'Washington',
        'state': 'DC',
        'display': 'Washington, D.C.',
        'latitude': 38.9072,
        'longitude': -77.0369,
        'default_zoom': 12,
        'property_tax_rate': 0.0085,      # DC moderate
        'insurance_rate': 0.004,          # DC low-moderate
        'str_permit_annual': 100,         # DC STR license
        'cleaning_per_turnover': 175,     # very high COL
        'avg_stay_nights': 3.0,           # government + tourism
        'base_utilities_monthly': 250,    # hot summers, cold winters
        'ia_state': 'dc',
        'ia_city': 'washington-dc',
        'ia_date': '2025-09-22',
        'redfin_region_id': 12839,
        'assessor': None,
    },
    'boston': {
        'name': 'Boston',
        'state': 'MA',
        'display': 'Boston, MA',
        'latitude': 42.3601,
        'longitude': -71.0589,
        'default_zoom': 12,
        'property_tax_rate': 0.0106,      # MA moderate-high
        'insurance_rate': 0.005,          # MA moderate — nor'easters
        'str_permit_annual': 200,         # Boston STR registration
        'cleaning_per_turnover': 165,     # very high COL
        'avg_stay_nights': 3.0,           # university + tourism
        'base_utilities_monthly': 275,    # cold winters, heating heavy
        'ia_state': 'ma',
        'ia_city': 'boston',
        'ia_date': '2025-09-23',
        'redfin_region_id': 1826,
        'assessor': None,
    },
    # san-francisco: removed — Dec 2025 IA scrape has zero price data
    'las-vegas': {
        'name': 'Las Vegas',
        'state': 'NV',
        'display': 'Las Vegas, NV',
        'latitude': 36.1699,
        'longitude': -115.1398,
        'default_zoom': 11,
        'property_tax_rate': 0.006,       # NV low — no state income tax
        'insurance_rate': 0.003,          # NV very low risk
        'str_permit_annual': 1000,        # Clark County STR business license
        'cleaning_per_turnover': 120,     # mid COL
        'avg_stay_nights': 3.0,           # entertainment weekend
        'base_utilities_monthly': 300,    # extreme heat, AC very heavy
        'ia_state': 'nv',
        'ia_city': 'clark-county-nv',
        'ia_date': '2025-09-23',
        'redfin_region_id': 10201,
        'assessor': None,
    },
    'fort-lauderdale': {
        'name': 'Fort Lauderdale',
        'state': 'FL',
        'display': 'Fort Lauderdale, FL',
        'latitude': 26.1224,
        'longitude': -80.1373,
        'default_zoom': 12,
        'property_tax_rate': 0.011,       # FL moderate
        'insurance_rate': 0.008,          # FL high — hurricane risk
        'str_permit_annual': 250,         # Broward County STR license
        'cleaning_per_turnover': 130,     # mid COL Florida
        'avg_stay_nights': 5.0,           # beach / snowbird
        'base_utilities_monthly': 275,    # hot, humid, AC heavy
        'ia_state': 'fl',
        'ia_city': 'broward-county',
        'ia_date': '2025-09-26',
        'redfin_region_id': 6173,
        'assessor': None,
    },
    'fort-worth': {
        'name': 'Fort Worth',
        'state': 'TX',
        'display': 'Fort Worth, TX',
        'latitude': 32.7555,
        'longitude': -97.3308,
        'default_zoom': 11,
        'property_tax_rate': 0.018,       # TX high property tax
        'insurance_rate': 0.007,          # TX highest — storms, hail
        'str_permit_annual': 0,           # Fort Worth no STR permit required
        'cleaning_per_turnover': 95,      # low COL
        'avg_stay_nights': 2.5,           # business / stockyards
        'base_utilities_monthly': 275,    # hot summers
        'ia_state': 'tx',
        'ia_city': 'fort-worth',
        'ia_date': '2025-09-16',
        'redfin_region_id': 30827,
        'assessor': None,
    },
    'honolulu': {
        'name': 'Honolulu',
        'state': 'HI',
        'display': 'Honolulu, HI',
        'latitude': 21.3069,
        'longitude': -157.8583,
        'default_zoom': 12,
        'property_tax_rate': 0.003,       # HI very low property tax
        'insurance_rate': 0.004,          # HI moderate — hurricane + flood
        'str_permit_annual': 500,         # Honolulu NUC permit
        'cleaning_per_turnover': 175,     # island premium, high COL
        'avg_stay_nights': 5.5,           # beach vacation
        'base_utilities_monthly': 250,    # moderate climate, some AC
        'ia_state': 'hi',
        'ia_city': 'hawaii',
        'ia_date': '2025-09-16',
        'redfin_region_id': 34945,
        'assessor': None,
    },
}

# ---------------------------------------------------------------------------
# Seasonal adjustment factors per city (index 0 = Jan, 11 = Dec).
# Each array sums to 12.0 (averages to 1.0) so annual totals are preserved.
# Based on documented STR market patterns for each metro.
# ---------------------------------------------------------------------------
SEASONAL_FACTORS = {
    'austin': {
        # SXSW (Mar) + ACL/F1 (Oct) peaks; hot summers dip
        'price':     [0.82, 0.88, 1.35, 1.10, 1.05, 0.95, 0.88, 0.85, 0.92, 1.25, 1.00, 0.95],
        'occupancy': [0.80, 0.85, 1.25, 1.10, 1.08, 1.00, 0.90, 0.88, 0.95, 1.22, 1.02, 0.95],
    },
    'nashville': {
        # Summer tourism + CMA Fest (Jun); winter lull
        'price':     [0.75, 0.78, 0.90, 1.05, 1.15, 1.25, 1.20, 1.10, 1.05, 1.12, 0.85, 0.80],
        'occupancy': [0.72, 0.78, 0.88, 1.05, 1.15, 1.22, 1.20, 1.12, 1.05, 1.12, 0.85, 0.86],
    },
    'san-diego': {
        # Classic beach city: strong summer peak, mild winters
        'price':     [0.80, 0.82, 0.90, 0.95, 1.05, 1.20, 1.30, 1.25, 1.10, 0.98, 0.85, 0.80],
        'occupancy': [0.78, 0.82, 0.90, 0.95, 1.08, 1.18, 1.28, 1.22, 1.10, 0.98, 0.85, 0.86],
    },
    'dallas': {
        # Business travel market — flatter seasonality; spring/fall slight peaks
        'price':     [0.90, 0.92, 1.08, 1.10, 1.06, 1.00, 0.92, 0.90, 0.96, 1.12, 1.05, 0.99],
        'occupancy': [0.88, 0.90, 1.05, 1.08, 1.05, 1.02, 0.95, 0.92, 0.98, 1.10, 1.02, 1.05],
    },
    'asheville': {
        # Strong fall foliage peak (Oct); summer tourism; deep winter lull
        'price':     [0.72, 0.75, 0.82, 0.95, 1.08, 1.18, 1.22, 1.15, 1.10, 1.32, 0.90, 0.81],
        'occupancy': [0.70, 0.72, 0.82, 0.95, 1.10, 1.20, 1.22, 1.18, 1.12, 1.28, 0.90, 0.81],
    },
    'seattle': {
        # Pacific NW summer peak (Jul-Aug); rainy winter lull
        'price':     [0.78, 0.80, 0.88, 0.95, 1.08, 1.20, 1.28, 1.25, 1.10, 0.95, 0.85, 0.88],
        'occupancy': [0.75, 0.78, 0.85, 0.95, 1.10, 1.22, 1.28, 1.25, 1.08, 0.95, 0.82, 0.97],
    },
    'denver': {
        # Ski season (Dec-Mar) + summer outdoor peaks; shoulder dips
        'price':     [1.15, 1.10, 1.20, 0.95, 0.90, 1.05, 1.15, 1.10, 0.95, 0.88, 0.82, 0.75],
        'occupancy': [1.10, 1.08, 1.15, 0.95, 0.92, 1.08, 1.18, 1.12, 0.95, 0.88, 0.80, 0.79],
    },
    'new-orleans': {
        # Mardi Gras (Feb) + Jazz Fest (Apr-May) peaks; brutal summer heat lull
        'price':     [0.90, 1.35, 1.05, 1.20, 1.10, 0.82, 0.78, 0.75, 0.80, 1.05, 1.00, 1.20],
        'occupancy': [0.88, 1.30, 1.02, 1.18, 1.08, 0.82, 0.78, 0.75, 0.82, 1.05, 1.02, 1.30],
    },
    'washington-dc': {
        # Cherry blossom (Mar-Apr) + summer tourism; winter dip
        'price':     [0.82, 0.85, 1.15, 1.20, 1.10, 1.08, 1.05, 0.95, 1.00, 1.05, 0.88, 0.87],
        'occupancy': [0.80, 0.85, 1.12, 1.18, 1.10, 1.08, 1.05, 0.98, 1.02, 1.05, 0.88, 0.89],
    },
    'boston': {
        # Fall foliage (Oct) + summer; deep winter lull
        'price':     [0.75, 0.78, 0.88, 1.00, 1.10, 1.18, 1.22, 1.15, 1.10, 1.25, 0.82, 0.77],
        'occupancy': [0.72, 0.78, 0.88, 1.02, 1.10, 1.20, 1.22, 1.15, 1.10, 1.22, 0.82, 0.79],
    },
    'las-vegas': {
        # Convention season (Jan, Mar) + NYE/holiday; brutal summer heat
        'price':     [1.15, 1.00, 1.20, 1.05, 0.95, 0.82, 0.78, 0.80, 0.90, 1.05, 1.10, 1.20],
        'occupancy': [1.12, 1.00, 1.18, 1.05, 0.95, 0.82, 0.78, 0.80, 0.92, 1.05, 1.10, 1.23],
    },
    'fort-lauderdale': {
        # Snowbird season (Jan-Mar) peak; hurricane summer dip
        'price':     [1.25, 1.30, 1.25, 1.05, 0.90, 0.82, 0.80, 0.78, 0.75, 0.85, 1.05, 1.20],
        'occupancy': [1.22, 1.28, 1.22, 1.05, 0.90, 0.82, 0.80, 0.78, 0.75, 0.85, 1.08, 1.25],
    },
    'fort-worth': {
        # Similar to Dallas — business travel; spring/fall slight peaks
        'price':     [0.90, 0.92, 1.08, 1.10, 1.06, 1.00, 0.92, 0.90, 0.96, 1.12, 1.05, 0.99],
        'occupancy': [0.88, 0.90, 1.05, 1.08, 1.05, 1.02, 0.95, 0.92, 0.98, 1.10, 1.02, 1.05],
    },
    'honolulu': {
        # Winter escape (Dec-Mar) peak; slight summer dip but always strong
        'price':     [1.15, 1.12, 1.10, 1.00, 0.92, 0.88, 0.90, 0.92, 0.88, 0.95, 1.05, 1.13],
        'occupancy': [1.12, 1.10, 1.08, 1.00, 0.92, 0.90, 0.92, 0.95, 0.90, 0.95, 1.05, 1.11],
    },
}

# Pre-compute lng_factor for distance calculations
for _key, _city in CITIES.items():
    _city['lng_factor'] = 111 * math.cos(math.radians(_city['latitude']))

DEFAULT_CITY = 'austin'


def get_city(key):
    """Return city dict or raise KeyError."""
    if key not in CITIES:
        raise KeyError(f"Unknown city: {key!r}. Available: {list(CITIES.keys())}")
    return CITIES[key]


def get_city_or_default(key):
    """Return city dict, falling back to the default city."""
    return CITIES.get(key, CITIES[DEFAULT_CITY])


def get_seasonal_factors(key):
    """Return seasonal factors for a city, or flat (1.0) if unknown."""
    flat = {'price': [1.0] * 12, 'occupancy': [1.0] * 12}
    return SEASONAL_FACTORS.get(key or DEFAULT_CITY, flat)


def list_cities():
    """Return list of {key, name, state, display, ...} for all cities."""
    result = []
    for key, city in CITIES.items():
        result.append({'key': key, **city})
    return result
