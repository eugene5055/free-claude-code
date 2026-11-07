"""Which providers run locally (llama.cpp, LM Studio, Ollama or custom providers on this machine or the LAN).

The executor uses this so a request for a local model never switches to a cloud model by itself: automatic
fallbacks (provider_health.proven_models) of a request whose configured candidates are all local stay local.
"""

import ipaddress
from urllib.parse import urlsplit

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings


def is_local_url(url: str | None) -> bool:
    """True for loopback, private (RFC 1918 / ULA), link-local, localhost and mDNS (*.local) hosts."""
    host = (urlsplit(url or "").hostname or "").rstrip(".").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith((".localhost", ".local")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def local_provider_ids(settings: Settings | None) -> frozenset[str]:
    """Ids of configured providers whose base URL is local (custom providers and built-in local ones)."""
    if settings is None:
        return frozenset()
    local = {
        item.provider_id
        for item in settings.custom_providers
        if is_local_url(item.base_url)
    }
    for provider_id, descriptor in PROVIDER_CATALOG.items():
        base_url = descriptor.default_base_url
        if descriptor.base_url_attr:
            value = getattr(settings, descriptor.base_url_attr, None)
            if isinstance(value, str) and value:
                base_url = value
        if is_local_url(base_url):
            local.add(provider_id)
    return frozenset(local)
