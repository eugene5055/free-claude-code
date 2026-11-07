"""llama-server slot scheduling: priority admission, keep-alives, KV save/restore."""

import asyncio
import json
from typing import Any

import httpx
import pytest

from free_claude_code.providers import llama_slots
from free_claude_code.providers.llama_slots import (
    LlamaSlotScheduler,
    RequestClass,
    RequestProfile,
    SlotOptions,
    apply_body_extras,
    classify,
    conversation_key,
    message_chain,
    restore_worthwhile,
    thinking_extras,
)

MAIN_HEADERS = {"X-Claude-Code-Session-Id": "7bc16e39-ca04"}
SUB_HEADERS = {**MAIN_HEADERS, "x-claude-code-agent-id": "a7dfdff4099f96ab0"}
DUO_HEADERS = {"X-Claude-Code-Session-Id": "d0d0d0d0-1234-5678"}


class FakeLlama:
    def __init__(self, *, slots=1, save_path=True, props_status=200):
        self.slots = slots
        self.save_path = save_path
        self.props_status = props_status
        self.calls: list[tuple[str, str, str]] = []
        self.files: dict[str, int] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            if self.props_status != 200:
                return httpx.Response(self.props_status, text="not found")
            return httpx.Response(
                200, json={"total_slots": self.slots, "model_path": "/m/q8.gguf"}
            )
        if request.url.path.startswith("/slots/"):
            action = request.url.params["action"]
            name = json.loads(request.content)["filename"]
            self.calls.append((action, request.url.path.rsplit("/", 1)[1], name))
            if not self.save_path:
                return httpx.Response(
                    501,
                    json={
                        "error": {
                            "message": "This server does not support slots action."
                        }
                    },
                )
            if action == "save":
                self.files[name] = 1
                return httpx.Response(
                    200, json={"n_saved": 50000, "n_written": 10 * 1024 * 1024}
                )
            return httpx.Response(200, json={"n_restored": 50000, "n_read": 1})
        return httpx.Response(404)


def scheduler(fake: FakeLlama, **options) -> LlamaSlotScheduler:
    return LlamaSlotScheduler(
        "http://llama.test/v1",
        options=SlotOptions(**options),
        transport=httpx.MockTransport(fake.handler),
    )


def profile(klass, conv, chain=("h", "m1"), tokens=50000, extras=None):
    return RequestProfile(
        klass=klass,
        conversation=conv,
        chain=tuple(chain),
        input_tokens=tokens,
        extras=extras or {},
    )


def test_classify_uses_claude_code_markers():
    common: dict[str, Any] = {
        "requested_model": "claude-sonnet-4-5",
        "system_text": "",
        "has_tools": True,
    }
    assert classify(MAIN_HEADERS, is_classifier=False, **common) is RequestClass.MAIN
    assert classify(SUB_HEADERS, is_classifier=False, **common) is RequestClass.SUBAGENT
    marked: dict[str, Any] = {
        **common,
        "system_text": "cc_version=2.1; cc_is_subagent=true;",
    }
    assert (
        classify(MAIN_HEADERS, is_classifier=False, **marked) is RequestClass.SUBAGENT
    )
    assert classify(DUO_HEADERS, is_classifier=False, **common) is RequestClass.DUO
    assert classify(DUO_HEADERS, is_classifier=True, **common) is RequestClass.DUO
    assert classify(MAIN_HEADERS, is_classifier=True, **common) is RequestClass.MAIN
    title: dict[str, Any] = {
        **common,
        "requested_model": "claude-haiku-4-5",
        "has_tools": False,
    }
    assert (
        classify(MAIN_HEADERS, is_classifier=False, **title) is RequestClass.BACKGROUND
    )
    explore: dict[str, Any] = {**common, "requested_model": "claude-haiku-4-5"}
    assert (
        classify(SUB_HEADERS, is_classifier=False, **explore) is RequestClass.SUBAGENT
    )


def test_chain_ignores_moving_cache_control_and_extends():
    first = message_chain(
        {"system": "s"},
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "a",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )
    later = message_chain(
        {"system": "s"},
        [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
    )
    assert later[: len(first)] == first
    assert restore_worthwhile(first, later)
    other = message_chain({"system": "different"}, [{"role": "user", "content": "a"}])
    assert not restore_worthwhile(first, other)


def test_conversation_key_separates_agents_and_sessions():
    chain = ("head0000000000", "m1")
    assert conversation_key(MAIN_HEADERS, chain) != conversation_key(SUB_HEADERS, chain)
    assert conversation_key({}, chain) == "anon:m1"
    assert conversation_key({}, ("only",)) is None


def test_thinking_extras_leave_main_and_cap_others():
    opts = SlotOptions(subagent_thinking_budget=4096, low_effort_thinking_budget=1024)
    kw: dict[str, Any] = {"thinking_off": False, "options": opts, "max_tokens": 32000}
    assert thinking_extras(RequestClass.MAIN, effort="high", **kw) == {}
    assert thinking_extras(RequestClass.MAIN, effort="low", **kw) == {}
    assert thinking_extras(RequestClass.SUBAGENT, effort="high", **kw) == {
        "thinking_budget_tokens": 4096
    }
    assert thinking_extras(RequestClass.SUBAGENT, effort="low", **kw) == {
        "thinking_budget_tokens": 1024
    }
    assert thinking_extras(RequestClass.DUO, effort="low", **kw) == {
        "thinking_budget_tokens": 1024
    }
    assert thinking_extras(RequestClass.DUO, effort="high", **kw) == {}
    off: dict[str, Any] = {**kw, "thinking_off": True}
    background = thinking_extras(RequestClass.BACKGROUND, effort=None, **kw)
    assert background == {"thinking_budget_tokens": 0, "max_tokens_cap": 4096}
    assert thinking_extras(RequestClass.SUBAGENT, effort="high", **off) == {}


@pytest.mark.asyncio
async def test_body_extras_pin_slot_and_cap_tokens_only_inside_stream():
    body = {"max_tokens": 32000, "reasoning_effort": "high"}
    await apply_body_extras(body)
    assert body == {"max_tokens": 32000, "reasoning_effort": "high"}
    fake = FakeLlama(slots=2)
    sched = scheduler(fake)
    seen = {}

    async def inner():
        prepared = {"max_tokens": 32000, "reasoning_effort": "high"}
        await apply_body_extras(prepared)
        seen.update(prepared)
        yield "frame"

    extras = {"thinking_budget_tokens": 0, "max_tokens_cap": 4096}
    frames = [
        f
        async for f in sched.stream(
            profile(RequestClass.BACKGROUND, "c", extras=extras),
            inner,
            keepalive="ka",
        )
    ]
    assert frames == ["frame"]
    assert seen["extra_body"]["id_slot"] in (0, 1)
    assert seen["max_tokens"] == 4096
    assert seen["extra_body"]["thinking_budget_tokens"] == 0
    assert "id_slot" not in seen  # never an OpenAI SDK kwarg
    assert seen["reasoning_effort"] == "high"  # the client's level is still sent
    await sched.aclose()


@pytest.mark.asyncio
async def test_queue_grants_by_priority_and_never_exceeds_slots():
    fake = FakeLlama(slots=1)
    sched = scheduler(fake)
    holder = await sched.acquire(profile(RequestClass.MAIN, "busy"))
    order = []

    async def wait(klass, conv):
        lease = await sched.acquire(profile(klass, conv))
        order.append(klass)
        await asyncio.sleep(0)
        sched.release(lease)

    tasks = [
        asyncio.create_task(wait(RequestClass.BACKGROUND, "bg")),
        asyncio.create_task(wait(RequestClass.DUO, "duo")),
        asyncio.create_task(wait(RequestClass.SUBAGENT, "sub")),
        asyncio.create_task(wait(RequestClass.MAIN, "main")),
    ]
    await asyncio.sleep(0.05)
    assert order == []  # one slot, held
    sched.release(holder)
    await asyncio.gather(*tasks)
    assert order == [
        RequestClass.MAIN,
        RequestClass.SUBAGENT,
        RequestClass.DUO,
        RequestClass.BACKGROUND,
    ]
    await sched.aclose()


@pytest.mark.asyncio
async def test_waiters_age_so_background_is_not_starved(monkeypatch):
    now = [1000.0]
    fake = FakeLlama(slots=1)
    sched = LlamaSlotScheduler(
        "http://llama.test/v1",
        options=SlotOptions(),
        transport=httpx.MockTransport(fake.handler),
        clock=lambda: now[0],
    )
    holder = await sched.acquire(profile(RequestClass.MAIN, "busy"))
    bg = asyncio.create_task(sched.acquire(profile(RequestClass.BACKGROUND, "bg")))
    await asyncio.sleep(0.01)
    now[0] += 4 * llama_slots.AGE_SECONDS
    main = asyncio.create_task(sched.acquire(profile(RequestClass.MAIN, "m")))
    await asyncio.sleep(0.01)
    sched.release(holder)
    await asyncio.sleep(0.01)
    assert bg.done() and not main.done()
    sched.release(bg.result())
    sched.release(await main)
    await sched.aclose()


@pytest.mark.asyncio
async def test_queued_stream_emits_keepalives_until_granted():
    fake = FakeLlama(slots=1)
    sched = scheduler(fake)
    holder = await sched.acquire(profile(RequestClass.MAIN, "busy"))

    async def inner():
        yield "data"

    frames = []

    async def consume():
        record = frames.append
        async for frame in sched.stream(
            profile(RequestClass.SUBAGENT, "sub"), inner, keepalive="KA", interval=0.01
        ):
            record(frame)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.08)
    assert frames and set(frames) == {"KA"}
    sched.release(holder)
    await task
    assert frames[-1] == "data"
    await sched.aclose()


@pytest.mark.asyncio
async def test_silent_upstream_gets_keepalives_and_slot_is_released():
    fake = FakeLlama(slots=1)
    sched = scheduler(fake)

    async def slow():
        await asyncio.sleep(0.05)
        yield "late"

    frames = [
        f
        async for f in sched.stream(
            profile(RequestClass.MAIN, "m"), slow, keepalive="KA", interval=0.01
        )
    ]
    assert frames[-1] == "late" and "KA" in frames
    again = await asyncio.wait_for(sched.acquire(profile(RequestClass.MAIN, "m")), 1)
    sched.release(again)
    await sched.aclose()


@pytest.mark.asyncio
async def test_cancelled_waiter_leaves_queue_and_closing_stream_releases():
    fake = FakeLlama(slots=1)
    sched = scheduler(fake)
    holder = await sched.acquire(profile(RequestClass.MAIN, "busy"))
    waiter = asyncio.create_task(sched.acquire(profile(RequestClass.DUO, "d")))
    await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    sched.release(holder)

    async def endless():
        while True:
            yield "x"
            await asyncio.sleep(0)

    stream = sched.stream(profile(RequestClass.MAIN, "m"), endless, keepalive="KA")
    assert await stream.__anext__() == "x"
    await stream.aclose()
    lease = await asyncio.wait_for(sched.acquire(profile(RequestClass.MAIN, "n")), 1)
    assert lease.slot == 0
    sched.release(lease)
    await sched.aclose()


@pytest.mark.asyncio
async def test_switching_conversations_saves_then_restores_matching_prefix():
    fake = FakeLlama(slots=1)
    sched = scheduler(fake)
    a1 = ("hA", "a1")
    lease = await sched.acquire(profile(RequestClass.MAIN, "A", a1))
    assert lease.extras["id_slot"] == 0
    sched.release(lease)
    sched.release(
        await sched.acquire(profile(RequestClass.SUBAGENT, "B", ("hB", "b1")))
    )
    assert fake.calls == [("save", "0", "fcc-llama-0000.bin")]
    lease = await sched.acquire(profile(RequestClass.MAIN, "A", (*a1, "a2", "a3")))
    assert lease.restored
    assert fake.calls[-1] == ("restore", "0", "fcc-llama-0000.bin")
    assert ("save", "0", "fcc-llama-0001.bin") in fake.calls
    sched.release(lease)
    # Same conversation again: resident, no KV movement.
    before = len(fake.calls)
    sched.release(
        await sched.acquire(profile(RequestClass.MAIN, "A", (*a1, "a2", "a3", "a4")))
    )
    assert len(fake.calls) == before
    await sched.aclose()


@pytest.mark.asyncio
async def test_diverged_prefix_skips_restore_and_small_prompts_are_not_saved():
    fake = FakeLlama(slots=1)
    sched = scheduler(fake)
    sched.release(
        await sched.acquire(profile(RequestClass.MAIN, "A", ("hA", "a1", "a2", "a3")))
    )
    sched.release(
        await sched.acquire(profile(RequestClass.MAIN, "B", ("hB", "b1"), tokens=100))
    )
    lease = await sched.acquire(
        profile(RequestClass.MAIN, "A", ("hA", "x1", "x2", "x3"))
    )
    assert not lease.restored
    assert [c[0] for c in fake.calls] == ["save"]  # B (100 tokens) not saved
    sched.release(lease)
    await sched.aclose()


@pytest.mark.asyncio
async def test_lru_files_stay_within_budget_and_names_are_reused():
    fake = FakeLlama(slots=1)
    sched = scheduler(fake, save_budget_mb=25)  # two 10 MB files fit
    for conv in "ABCDE":
        sched.release(
            await sched.acquire(profile(RequestClass.MAIN, conv, (f"h{conv}", "m")))
        )
    names = {call[2] for call in fake.calls if call[0] == "save"}
    assert len(names) <= 3
    assert len(sched._saves) == 2
    await sched.aclose()


@pytest.mark.asyncio
async def test_without_slot_save_path_requests_are_still_pinned():
    fake = FakeLlama(slots=1, save_path=False)
    sched = scheduler(fake)
    sched.release(await sched.acquire(profile(RequestClass.MAIN, "A")))
    lease = await sched.acquire(profile(RequestClass.MAIN, "B", ("hB", "b")))
    assert lease.extras["id_slot"] == 0 and not lease.restored
    sched.release(lease)
    calls = len(fake.calls)
    sched.release(await sched.acquire(profile(RequestClass.MAIN, "C", ("hC", "c"))))
    assert len(fake.calls) == calls  # not retried until the recheck interval
    await sched.aclose()


@pytest.mark.asyncio
async def test_non_llama_server_passes_through_unpinned():
    fake = FakeLlama(props_status=404)
    sched = scheduler(fake)
    leases = [await sched.acquire(profile(RequestClass.MAIN, str(i))) for i in range(5)]
    assert all(lease.slot is None and "id_slot" not in lease.extras for lease in leases)
    await sched.aclose()


@pytest.mark.asyncio
async def test_thinking_cap_composes_with_existing_exact_budget():
    token = llama_slots._BODY_EXTRAS.set({"thinking_budget_tokens": 4096})
    try:
        # The SDK merges extra_body over the request, so extra_body is what
        # llama-server sees; every case must end with the tightest budget there.
        tighter_top: dict[str, Any] = {"thinking_budget_tokens": 512}
        await apply_body_extras(tighter_top)
        assert tighter_top["thinking_budget_tokens"] == 512
        assert tighter_top["extra_body"]["thinking_budget_tokens"] == 512
        looser: dict[str, Any] = {"extra_body": {"thinking_budget_tokens": 16384}}
        await apply_body_extras(looser)
        assert looser["extra_body"]["thinking_budget_tokens"] == 4096
        client: dict[str, Any] = {"extra_body": {"thinking_budget_tokens": 256, "x": 1}}
        await apply_body_extras(client)
        assert client["extra_body"] == {"thinking_budget_tokens": 256, "x": 1}
        bigger_top: dict[str, Any] = {"thinking_budget_tokens": 9000}
        await apply_body_extras(bigger_top)
        assert bigger_top["thinking_budget_tokens"] == 4096
        assert bigger_top["extra_body"]["thinking_budget_tokens"] == 4096
        off: dict[str, Any] = {"extra_body": {"thinking_budget_tokens": 0}}
        await apply_body_extras(off)
        assert off["extra_body"]["thinking_budget_tokens"] == 0
        unset: dict[str, Any] = {"extra_body": {"chat_template_kwargs": {}}}
        await apply_body_extras(unset)
        assert unset["extra_body"] == {
            "chat_template_kwargs": {},
            "thinking_budget_tokens": 4096,
        }
        assert "thinking_budget_tokens" not in unset
    finally:
        llama_slots._BODY_EXTRAS.reset(token)


def test_auto_scheduling_targets_local_servers_only():
    assert llama_slots.is_local_url("http://127.0.0.1:11437/v1")
    assert llama_slots.is_local_url("http://localhost:8080/v1")
    assert llama_slots.is_local_url("http://169.254.226.211:11436/v1")
    assert llama_slots.is_local_url("http://192.168.68.58:11437/v1")
    assert not llama_slots.is_local_url("https://custom.example/team/api")
    assert not llama_slots.is_local_url("https://8.8.8.8/v1")


@pytest.mark.asyncio
async def test_custom_provider_pins_slot_and_caps_subagent_thinking_on_the_wire():
    import httpx2

    from free_claude_code.core.anthropic import MessagesRequest
    from free_claude_code.core.reasoning import (
        ReasoningControl,
        ReasoningEffort,
        ReasoningPolicy,
    )
    from tests.providers.test_custom_provider import completion, definition, provider

    bodies: list[dict[str, Any]] = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            text=completion("openai_chat"),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format="openai_chat", base_url="http://127.0.0.1:9/v1"),
        openai_handler=reply,
    )
    slots = instance._slots
    assert slots is not None
    await slots._http.aclose()
    slots._http = httpx.AsyncClient(
        transport=httpx.MockTransport(FakeLlama(slots=2).handler)
    )
    request = MessagesRequest(
        model="m", max_tokens=8000, messages=[{"role": "user", "content": "Hi"}]
    )
    policy = ReasoningPolicy(control=ReasoningControl.ON, effort=ReasoningEffort.HIGH)
    try:
        for headers in (MAIN_HEADERS, SUB_HEADERS):
            stream = instance.stream_messages(
                request, input_tokens=10, request_headers=headers, reasoning=policy
            )
            assert [frame async for frame in stream]
        main, sub = bodies
        assert main["id_slot"] in (0, 1) and "thinking_budget_tokens" not in main
        assert sub["id_slot"] in (0, 1) and sub["thinking_budget_tokens"] == 4096
    finally:
        await instance.cleanup()


def test_remote_custom_provider_is_not_scheduled():
    from tests.providers.test_custom_provider import definition, provider

    remote = provider(
        definition(api_format="openai_chat"), openai_handler=lambda r: None
    )
    assert remote._slots is None
    forced = provider(
        definition(
            api_format="openai_chat",
            slot_scheduling="off",
            base_url="http://127.0.0.1:9/v1",
        ),
        openai_handler=lambda r: None,
    )
    assert forced._slots is None
