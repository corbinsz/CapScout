"""Choose the for-sale provider for this process.

RentCast wins whenever an API key is configured. The Redfin CSV is used only
when ALLOW_REDFIN_FALLBACK=1 and no RentCast key is set. There is no silent
fallback from a failed licensed call onto an unlicensed scrape.
"""
import config
from services.providers.redfin_provider import RedfinProvider
from services.providers.rentcast_provider import RentCastProvider


def rentcast_configured():
    return bool(config.RENTCAST_API_KEY)


def active_provider():
    if rentcast_configured():
        return RentCastProvider()
    if config.ALLOW_REDFIN_FALLBACK:
        return RedfinProvider()
    return None


def provider_status():
    provider = active_provider()
    if provider is None:
        return {
            'source': None,
            'label': None,
            'licensed': False,
            'message': (
                'No licensed listing feed is configured. Add RENTCAST_API_KEY on the server. '
                'For local development only, ALLOW_REDFIN_FALLBACK=1 reads a previously saved Redfin CSV.'
            ),
        }
    return {
        'source': provider.name,
        'label': provider.label,
        'licensed': provider.licensed,
        'message': provider.notice,
    }
