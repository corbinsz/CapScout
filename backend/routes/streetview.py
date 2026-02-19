"""
Streetview image proxy route.
Serves cached Street View images from disk, fetching on demand from Google
if not yet cached. The API key never reaches the frontend.
"""
import os
import re

from flask import Blueprint, send_file, abort

import config
from services.streetview_service import get_streetview_image, _make_hash

streetview_bp = Blueprint('streetview', __name__)

# Map hash back to lat/lng isn't possible (one-way hash), so the route
# accepts lat/lng as query params for the initial fetch, and the hash
# is just for cache lookup after that.
# Alternative: encode lat/lng in the URL path directly.

# We use a simple pattern: /api/streetview/<lat>,<lng>.jpg
# The hash is computed server-side for cache storage.


@streetview_bp.route('/streetview/<image_hash>.jpg')
def serve_streetview(image_hash):
    """Serve a cached Street View image, fetching from Google on first request."""
    if not re.match(r'^[a-f0-9]{1,32}$', image_hash):
        abort(404)

    cache_dir = config.streetview_cache_dir()
    path = os.path.join(cache_dir, f'{image_hash}.jpg')

    # Already cached — serve immediately
    if os.path.isfile(path):
        return send_file(path, mimetype='image/jpeg', max_age=31536000)

    # Not cached — we need lat/lng to fetch. Check query params.
    from flask import request
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)

    if lat is None or lng is None:
        abort(404)

    # Verify the hash matches so clients can't request arbitrary coordinates
    if _make_hash(lat, lng) != image_hash:
        abort(404)

    # Fetch and cache
    result = get_streetview_image(lat, lng)
    if result is None:
        abort(404)

    return send_file(result, mimetype='image/jpeg', max_age=31536000)
