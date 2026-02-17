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
        'expense_ratio': 0.35,            # moderate COL
        'str_permit_annual': 0,           # Austin Type 2 STR permit is one-time, no annual fee
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
        'expense_ratio': 0.33,            # lower COL
        'str_permit_annual': 350,         # Nashville STR permit ~$313-350/yr
        'ia_state': 'tn',
        'ia_city': 'nashville',
        'ia_date': '2025-06-19',
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
        'expense_ratio': 0.40,            # high COL — cleaning, maintenance, mgmt all pricier
        'str_permit_annual': 1000,        # SD TOT registration + compliance costs
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
        'expense_ratio': 0.34,            # moderate COL
        'str_permit_annual': 0,           # Dallas no STR permit required
        'ia_state': 'tx',
        'ia_city': 'dallas',
        'ia_date': '2025-06-15',
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
        'expense_ratio': 0.32,            # low COL
        'str_permit_annual': 200,         # Buncombe County STR permit
        'ia_state': 'nc',
        'ia_city': 'asheville',
        'ia_date': '2025-06-17',
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
        'expense_ratio': 0.40,            # high COL — expensive labor market
        'str_permit_annual': 75,          # Seattle STR operator license
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
