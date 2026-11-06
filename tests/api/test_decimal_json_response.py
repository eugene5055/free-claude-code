"""Non-stream aggregates keep Decimal numbers parsed from SSE frames."""

import json

from free_claude_code.api.stream_frames import decimal_json_response, frame_data
from free_claude_code.core.chat_completions.response import ChatResponseAccumulator


def test_chat_aggregate_with_float_timings_serialises():
    aggregate = ChatResponseAccumulator()
    payload = frame_data(
        'data: {"id":"c1","object":"chat.completion.chunk","created":1,'
        '"model":"m","choices":[{"index":0,"delta":{"role":"assistant",'
        '"content":"hi"},"finish_reason":"stop"}],'
        '"timings":{"predicted_per_second":23.95}}'
    )
    aggregate.feed(payload)
    aggregate.finish()
    response = decimal_json_response(aggregate.result())
    body = json.loads(response.body)
    assert response.status_code == 200
    assert response.media_type == "application/json"
    assert body["timings"]["predicted_per_second"] == 23.95
    assert body["choices"][0]["message"]["content"] == "hi"
