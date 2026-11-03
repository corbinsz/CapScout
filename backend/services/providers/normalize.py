"""Shared helpers for turning a raw feed row into a CapScout listing."""
import hashlib
import math
import re


def safe_float(val, default=0.0):
    if val is None or val == '':
        return default
    try:
        number = float(str(val).replace(',', '').replace('$', '').strip())
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def safe_int(val, default=0):
    number = safe_float(val, default=None)
    if number is None:
        return default
    try:
        return int(number)
    except (TypeError, ValueError):
        return default


def listing_id(prefix, raw_key):
    """Stable URL-safe id. Raw feed ids sometimes contain commas and spaces."""
    raw = str(raw_key or '').strip()
    if re.fullmatch(r'[A-Za-z0-9_-]{1,64}', raw):
        return f'{prefix}-{raw}'
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]
    return f'{prefix}-{digest}'


def osm_tile_url(lat, lng):
    """Low-res map tile used only when a listing has no photo. Not a property photo."""
    if not lat or not lng:
        return None
    zoom = 17
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    lat_rad = math.radians(max(min(lat, 85.0), -85.0))
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return f'https://tile.openstreetmap.org/{zoom}/{x}/{y}.png'
