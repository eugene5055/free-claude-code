"""Authenticated fixed-route FCC council API; installed only after validation."""
import json
from fastapi import Body,Depends,HTTPException,Request
from .dependencies import get_services,get_settings,require_proxy_auth
from .ports import ApiServices
from free_claude_code.config.settings import Settings
from free_claude_code.application.routing import ModelRouter
from free_claude_code.application import model_usability
from .council_strict_route import StrictCouncilRouter,RoutingRefused,MODEL,RECIPIENT
POLICY='duo-council-nim-v1'
def guard(settings):
    if (settings.host!='127.0.0.1' or settings.port!=8082
            or settings.proxy_auth_enabled is not True
            or len(settings.proxy_auth_token)<32
            or not settings.nvidia_nim_api_keys
            or not model_usability.offered(MODEL)):
        raise HTTPException(503,'Council route prerequisites unavailable')
def metadata(services,settings):
    guard(settings)
    return {'status':'running','instance_id':services.admin.instance_id,
            'policy':POLICY,'model':MODEL,'recipient':RECIPIENT,
            'fallback_disabled':True,'authenticated_transport':True}
def register(router,create_response):
    @router.get('/v1/council/status')
    async def council_status(services:ApiServices=Depends(get_services),settings:Settings=Depends(get_settings),_auth=Depends(require_proxy_auth)):
        return metadata(services,settings)
    @router.post('/v1/council/messages')
    async def council_messages(request:Request,body:dict=Body(),services:ApiServices=Depends(get_services),settings:Settings=Depends(get_settings),_auth=Depends(require_proxy_auth)):
        state=metadata(services,settings)
        if (set(body)!={'expected_instance','policy','request'} or body['expected_instance']!=state['instance_id'] or body['policy']!=POLICY):
            raise HTTPException(409,'Council proxy instance or policy changed')
        raw=body['request']
        if (type(raw)!=dict or set(raw)!={'model','max_tokens','stream','messages','system'}
                or raw.get('model')!=MODEL or type(raw.get('max_tokens'))!=int
                or not 1<=raw['max_tokens']<=256 or raw.get('stream') is not False
                or type(raw.get('system'))!=str or type(raw.get('messages'))!=list
                or len(raw['messages'])!=1 or type(raw['messages'][0])!=dict
                or set(raw['messages'][0])!={'role','content'}
                or raw['messages'][0]['role']!='user' or type(raw['messages'][0]['content'])!=str
                or len(json.dumps(body).encode())>8192):
            raise HTTPException(400,'Bounded advisory request required')
        def factory(current):
            guard(current)
            return StrictCouncilRouter(ModelRouter(current),lambda p,m:p==RECIPIENT and m==MODEL and model_usability.offered(m))
        try:
            # Never forward the private local proxy token to an upstream provider.
            return await create_response(services,raw,request_id=request.headers.get('x-request-id','council'),request_headers=None,model_router_factory=factory)
        except RoutingRefused:
            raise HTTPException(409,'Fixed council route unavailable') from None
