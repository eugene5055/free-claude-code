"""Council-only FCC service: no general application, integrations, or cleanup."""
import asyncio,contextlib,os,pathlib,stat,uuid
from functools import partial
from types import SimpleNamespace
from fastapi import FastAPI,APIRouter
from free_claude_code.api.ports import ApiServices
from free_claude_code.api.council_api import guard,register
from free_claude_code.api.council_routes import _create_messages_response
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.paths import server_owner_lock_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.providers.runtime.runtime import ProviderRuntime,create_provider
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager

async def fixed_provider(provider_id,settings,admission,api_key_number=1):
    if provider_id!='nvidia_nim' or api_key_number!=1:
        raise ValueError('Council provider route refused')
    return await create_provider(provider_id,settings,admission,api_key_number)

def create_app(settings,*,manager_factory=ProviderRuntimeManager):
    guard(settings)
    settings=settings.model_copy(update={'log_raw_api_payloads':False,'log_raw_sse_events':False,'log_api_error_tracebacks':False})
    manager=manager_factory(settings,runtime_factory=partial(ProviderRuntime,provider_constructor=fixed_provider))
    @contextlib.asynccontextmanager
    async def lifespan(app):
        # Deliberately no standard FCC ApplicationRuntime.start(): no messaging,
        # integration discovery, code sessions, catalogs, or history cleanup.
        try:yield
        finally:
            manager.begin_shutdown()
            await asyncio.wait_for(manager.close(),10)
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    app.state.services=ApiServices(requests=manager,admin=SimpleNamespace(instance_id=uuid.uuid4().hex),tasks=None,web_tools=None,code=None)
    router=APIRouter();register(router,_create_messages_response);app.include_router(router)
    return app

def main():
    store=ManagedConfigStore()
    for path in (store.path.parent,store.path):
        s=path.lstat()
        if stat.S_ISLNK(s.st_mode) or s.st_uid!=os.getuid() or s.st_mode&0o077:
            raise RuntimeError('Private owned FCC configuration required')
    settings=store.read(env={}).settings;guard(settings)
    owner=InterprocessFileLock(server_owner_lock_path())
    if not owner.acquire():raise RuntimeError('Existing FCC owner; refusing duplicate service')
    try:
        import uvicorn
        uvicorn.run(create_app(settings),host='127.0.0.1',port=8082,access_log=False,
                    timeout_keep_alive=2,timeout_graceful_shutdown=10,limit_concurrency=2)
    finally:owner.release()

if __name__=='__main__':main()
