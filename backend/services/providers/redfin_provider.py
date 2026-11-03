"""Optional local-dev reader for a Redfin CSV the developer already saved.

This does not call Redfin. Live downloads of Redfin's unofficial map endpoint
are refused unless ALLOW_REDFIN_DOWNLOAD=1, and even then they are not part of
the commercial data path.
"""
from services.providers.base import empty_listing_message
from services.redfin_service import get_listings, has_data


class RedfinProvider:
    name = 'redfin'
    label = 'Redfin CSV (dev fallback)'
    licensed = False
    notice = (
        'These homes come from a saved file on this server, not a live licensed feed. '
        'The scores use the same model either way. A RentCast key switches the list to live listings.'
    )

    def empty_message(self, city_key):
        if not has_data(city_key):
            return empty_listing_message(self.name)
        return 'The saved Redfin file had no usable homes.'

    def listings(self, city_key):
        if not has_data(city_key):
            return []
        return get_listings(city_key)

    def get(self, city_key, property_id):
        if not str(property_id).startswith('redfin-'):
            return None
        for item in self.listings(city_key):
            if item['id'] == property_id:
                return item
        return None
