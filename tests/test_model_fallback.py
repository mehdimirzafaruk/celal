"""Fable is the main Mac model; when its usage limit is full Opus answers."""

import asyncio
from dataclasses import dataclass, field

import pytest
from claude_agent_sdk import AssistantMessage, RateLimitEvent, ResultMessage, TextBlock
from claude_agent_sdk.types import RateLimitInfo

from celal import brains
from celal.brains import MacClaudeBrain, looks_like_limit


def _result(text, *, error=False, status=None):
    return ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1, is_error=error, num_turns=1,
        session_id="s1", result=text, api_error_status=status,
    )


@dataclass
class FakeClient:
    opts: object
    script: dict
    sent: list = field(default_factory=list)
    _q: asyncio.Queue = field(default_factory=asyncio.Queue)

    async def connect(self):
        pass

    async def query(self, text):
        self.sent.append((self.opts.model, text))
        for m in self.script[self.opts.model]:
            await self._q.put(m)

    async def receive_messages(self):
        while True:
            yield await self._q.get()

    async def interrupt(self):
        pass

    async def disconnect(self):
        pass


@dataclass
class Opts:
    model: str | None
    fallback_model: str | None = None


def make(script, clients):
    def factory(opts):
        c = FakeClient(opts, script)
        clients.append(c)
        return c

    return MacClaudeBrain(
        cwd="~", model="fable", fallback_model="opus",
        client_factory=factory,
        options_factory=lambda **kw: Opts(kw["model"], kw.get("fallback_model")),
    )


@pytest.fixture(autouse=True)
def _clear_limits():
    brains._LIMIT_UNTIL.clear()
    yield
    brains._LIMIT_UNTIL.clear()


def test_looks_like_limit():
    assert looks_like_limit("You've hit your limit · resets 5pm")
    assert looks_like_limit("Claude AI usage limit reached|1761300000")
    assert looks_like_limit("", 429)
    assert not looks_like_limit("dosyada limit değişkeni var")


async def test_fable_answers_when_free():
    clients, out = [], []
    brain = make({"fable": [_result("selam")]}, clients)
    res = await brain.turn("merhaba", lambda k, v: out.append((k, v)))
    assert res.ok and res.text == "selam"
    assert clients[0].sent == [("fable", "merhaba")]
    assert clients[0].opts.fallback_model == "opus"


async def test_limit_switches_to_opus_and_retries():
    clients, out = [], []
    limit_msg = AssistantMessage(content=[TextBlock("You've hit your limit")], model="fable", error="rate_limit")
    script = {
        "fable": [limit_msg, _result("You've hit your limit", error=True, status=429)],
        "opus": [_result("opus cevabı")],
    }
    brain = make(script, clients)
    res = await brain.turn("merhaba", lambda k, v: out.append((k, v)))
    assert res.ok and res.text == "opus cevabı"
    assert [c.opts.model for c in clients] == ["fable", "opus"]
    assert not any(k == "text" and "limit" in v for k, v in out)  # notice is not spoken
    assert any(k == "progress" and "opus" in v for k, v in out)
    # a new tab starts directly on opus while fable is blocked
    clients2 = []
    brain2 = make(script, clients2)
    res2 = await brain2.turn("yeni", lambda k, v: None)
    assert res2.text == "opus cevabı" and [c.opts.model for c in clients2] == ["opus"]


async def test_rejected_rate_limit_event_triggers_fallback():
    clients = []
    ev = RateLimitEvent(RateLimitInfo(status="rejected", resets_at=None), uuid="u", session_id="s1")
    script = {"fable": [ev, _result("limit", error=True)], "opus": [_result("tamam")]}
    brain = make(script, clients)
    res = await brain.turn("x", lambda k, v: None)
    assert res.text == "tamam"
