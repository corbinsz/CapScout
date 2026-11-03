"""Request validation and a small in-process rate limiter.

No external store: this is enough to blunt accidental floods and credential
stuffing on a single Render worker. It is not a substitute for an edge
limiter if the API is put behind several machines.
"""
import re
import time
from functools import wraps

from flask import jsonify, request

from cities import CITIES

EMAIL_RE = re.compile(r'^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$', re.I)
PROPERTY_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,80}$')

_buckets: dict[tuple, list] = {}


def reset_rate_limits():
    _buckets.clear()


def client_ip():
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()[:64] or 'unknown'
    return (request.remote_addr or 'unknown')[:64]


def rate_limit(scope, limit, window=60):
    """Return 429 after `limit` hits per IP inside `window` seconds."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = (scope, client_ip())
            hits = [stamp for stamp in _buckets.get(key, []) if now - stamp < window]
            if len(hits) >= limit:
                response = jsonify({
                    'error': 'Too many requests. Please wait a moment and try again.',
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(window)
                return response
            hits.append(now)
            _buckets[key] = hits
            if len(_buckets) > 20000:
                cutoff = now - window
                stale = [k for k, stamps in _buckets.items() if not stamps or stamps[-1] < cutoff]
                for k in stale:
                    _buckets.pop(k, None)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def parse_city(value, default='austin'):
    """Return a registered city key, or None if the caller supplied garbage."""
    key = (value if value is not None else default)
    key = str(key).strip().lower()
    if key not in CITIES:
        return None
    return key


def city_or_400(value, default='austin'):
    key = parse_city(value, default=default)
    if key is None:
        return None, (jsonify({'error': 'Unknown city.'}), 400)
    return key, None


def bounded_float(raw, lo, hi):
    if raw is None or raw == '':
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError('not a number')
    if value != value or value in (float('inf'), float('-inf')):
        raise ValueError('not a finite number')
    if value < lo or value > hi:
        raise ValueError('out of range')
    return value


def bounded_int(raw, lo, hi):
    if raw is None or raw == '':
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError('not an integer')
    if value < lo or value > hi:
        raise ValueError('out of range')
    return value


def clean_text(value, max_len):
    if value is None:
        return ''
    text = str(value).replace('\x00', '').strip()
    text = ''.join(ch for ch in text if ch >= ' ' or ch == '\n')
    return text[:max_len]


def valid_email(value):
    email = clean_text(value, 254).lower()
    if not email or len(email) > 254 or not EMAIL_RE.match(email):
        return None
    return email


def valid_password(value):
    if not isinstance(value, str):
        return None
    if len(value) < 10 or len(value) > 128:
        return None
    if any(ord(ch) < 32 for ch in value):
        return None
    return value


def valid_property_id(value):
    if not value or not PROPERTY_ID_RE.match(str(value)):
        return None
    return str(value)
