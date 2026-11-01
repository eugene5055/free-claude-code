"""Server-side strict council router candidate; not installed or activated.

Wrap only the council-specific request path, never global ordinary-client routing.
The trusted server composition supplies the existing FCC ModelRouter and fresh
free-route eligibility. No model output can construct or change this policy.
"""
from dataclasses import replace

MODEL = 'nvidia_nim/z-ai/glm-5.3'
RECIPIENT = 'nvidia_nim'

class RoutingRefused(ValueError):
    pass

class StrictCouncilRouter:
    def __init__(self, router, free_eligible):
        self._router = router
        self._free_eligible = free_eligible

    def resolve(self, requested_model):
        if requested_model != MODEL or self._free_eligible(RECIPIENT, MODEL) is not True:
            raise RoutingRefused('fixed free route unavailable')
        route = self._router.resolve(MODEL)
        return self._strict(route)

    def _strict(self, route):
        if self._free_eligible(RECIPIENT, MODEL) is not True:
            raise RoutingRefused('fixed free route unavailable')
        target = route.primary
        if (target.provider_id != RECIPIENT or target.provider_model_ref != MODEL
                or target.provider_model != 'z-ai/glm-5.3' or target.api_key_number != 1):
            raise RoutingRefused('resolved recipient changed')
        # FCC ResolvedModelRoute is a frozen dataclass. Drop BOTH configured and
        # model-group sibling fallbacks after its real resolution, before dispatch.
        return replace(route, fallbacks=())

    def resolve_messages_request(self, request):
        # MessagesHandler resolves a second time. It must receive this wrapper,
        # not construct its ordinary router after the outer route check.
        self._request(request)
        routed = self._router.resolve_messages_request(request)
        return replace(routed, resolved=self._strict(routed.resolved))

    def route_native_messages(self, request, resolved):
        self._request(request)
        routed = self._router.route_native_messages(request, self._strict(resolved))
        return replace(routed, resolved=self._strict(routed.resolved))

    def _request(self, request):
        if (getattr(request, 'model', None) != MODEL
                or type(getattr(request, 'max_tokens', None)) is not int
                or not 1 <= request.max_tokens <= 256
                or getattr(request, 'stream', None) is not False
                or getattr(request, 'tools', None)
                or getattr(request, 'tool_choice', None)):
            raise RoutingRefused('non-advisory request')
