"""Normalized for-sale listing shape used by every provider."""


def empty_listing_message(provider_name):
    if provider_name == 'rentcast':
        return (
            'RentCast returned no active listings for this city. '
            'The response is cached so we do not repeat a billed API call.'
        )
    if provider_name == 'redfin':
        return 'No saved Redfin CSV for this city. This fallback is for local development only.'
    return 'No for-sale listings are available.'
