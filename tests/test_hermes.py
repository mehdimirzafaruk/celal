"""HermesClient against a mocked HTTP transport (no network)."""

import json

import httpx
import pytest

from celal.hermes import HermesClient, HermesError, SSEParser


def sse(*events):
    out = []
    for ev in events:
        out.append(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n")
    return "".join(out).encode()


RESPONSES_STREAM = sse(
    {"type": "response.created", "response": {"id": "r1"}},
    {"type": "response.output_item.added", "item": {"type": "function_call", "name": "web_search"}},
    {"type": "response.output_item.added", "item": {"type": "message"}},
    {"type": "response.output_text.delta", "delta": "Merhaba "},
    {"type": "response.output_text.delta", "delta": "Mirza."},
    {"type": "response.completed", "response": {"status": "completed"}},
)


def client_with(settings, handler):
    return HermesClient(settings, transport=httpx.MockTransport(handler))


async def collect(client, text):
    return [(e.kind, e.value) async for e in client.stream(text)]


async def test_responses_stream_sends_conversation_and_yields_text_and_tools(settings):
    seen = {}

    def handler(request: httpx.Request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=RESPONSES_STREAM, headers={"content-type": "text/event-stream"})

    client = client_with(settings, handler)
    events = await collect(client, "merhaba Celal")
    assert events == [("tool", "web_search"), ("text", "Merhaba "), ("text", "Mirza.")]
    assert seen["path"] == "/v1/responses"
    assert seen["auth"] == "Bearer test-secret-key"
    assert seen["body"]["conversation"] == "celal-mac"
    assert seen["body"]["stream"] is True and seen["body"]["input"] == "merhaba Celal"
    assert seen["body"]["model"] == "celal"
    await client.aclose()


async def test_new_conversation_changes_server_side_name(settings):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=RESPONSES_STREAM)

    client = client_with(settings, handler)
    name = client.new_conversation()
    await collect(client, "x")
    assert name.startswith("celal-mac-") and bodies[0]["conversation"] == name


async def test_chat_mode_keeps_history_client_side(settings):
    settings.api = "chat"
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        n = len(bodies)
        chunks = [
            f'data: {json.dumps({"choices": [{"delta": {"content": f"cevap {n}"}}]})}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(200, content="".join(chunks).encode())

    client = client_with(settings, handler)
    assert await collect(client, "bir") == [("text", "cevap 1")]
    assert await collect(client, "iki") == [("text", "cevap 2")]
    assert bodies[1]["stream"] is True
    roles = [(m["role"], m["content"]) for m in bodies[1]["messages"] if m["role"] != "system"]
    assert roles == [("user", "bir"), ("assistant", "cevap 1"), ("user", "iki")]


async def test_401_is_a_turkish_error_without_the_key(settings):
    client = client_with(settings, lambda r: httpx.Response(401, json={"error": "unauthorized"}))
    with pytest.raises(HermesError) as e:
        await collect(client, "x")
    assert "401" in str(e.value) and "test-secret-key" not in str(e.value)
    ok, reason = await client.ping()
    assert not ok and "401" in reason


async def test_unreachable_server(settings):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = client_with(settings, handler)
    with pytest.raises(HermesError, match="bağlanılamadı"):
        await collect(client, "x")
    ok, reason = await client.ping()
    assert not ok and "bağlanılamadı" in reason


async def test_response_failed_event_raises(settings):
    body = sse({"type": "response.failed", "response": {"error": {"message": "model down"}}})
    client = client_with(settings, lambda r: httpx.Response(200, content=body))
    with pytest.raises(HermesError, match="model down"):
        await collect(client, "x")


async def test_missing_api_key_file(settings, tmp_path):
    settings.api_key_file = str(tmp_path / "yok")
    client = client_with(settings, lambda r: httpx.Response(200, json={"data": []}))
    ok, reason = await client.ping()
    assert not ok and "API anahtarı yok" in reason


async def test_ping_ok(settings):
    client = client_with(settings, lambda r: httpx.Response(200, json={"data": [{"id": "celal"}]}))
    assert await client.ping() == (True, "bağlı")


def test_sse_parser_groups_multiline_data():
    p = SSEParser()
    assert p.feed("event: x") is None
    assert p.feed("data: a") is None
    assert p.feed("data: b") is None
    assert p.feed("") == ("x", "a\nb")
    assert p.feed("") is None
