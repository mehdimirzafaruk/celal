"""ClaudeAgentLLMService with a fake SDK client (no CLI, no network)."""

import asyncio

from claude_agent_sdk import ResultMessage, StreamEvent
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import SleepFrame, run_test

from jarvis.llm_claude import (
    ClaudeAgentLLMService,
    ClaudeSession,
    build_options,
    last_user_text,
    text_delta,
)


def delta(text: str) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id="s",
        event={"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    )


def result(terminal_reason: str = "completed") -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=5,
        is_error=False,
        num_turns=1,
        session_id="s",
        terminal_reason=terminal_reason,
    )


class FakeClient:
    """Mimics ClaudeSDKClient: query() then receive_response() until ResultMessage."""

    instances: list["FakeClient"] = []

    def __init__(self, options=None, replies=None, delay=0.0):
        self.options = options
        self.replies = list(replies or [])
        self.delay = delay
        self.queries: list[str] = []
        self.connects = 0
        self.interrupts = 0
        self.disconnected = False
        self._interrupted = asyncio.Event()
        FakeClient.instances.append(self)

    async def connect(self):
        self.connects += 1

    async def query(self, prompt):
        self.queries.append(prompt)
        self._interrupted.clear()

    async def receive_response(self):
        chunks = self.replies.pop(0)
        for chunk in chunks:
            if self._interrupted.is_set():
                break
            if self.delay:
                try:
                    await asyncio.wait_for(self._interrupted.wait(), timeout=self.delay)
                    break
                except TimeoutError:
                    pass
            yield delta(chunk)
        yield result("aborted_streaming" if self._interrupted.is_set() else "completed")

    async def interrupt(self):
        self.interrupts += 1
        self._interrupted.set()

    async def disconnect(self):
        self.disconnected = True


def make_service(**client_kwargs):
    holder = {}

    def factory(options):
        holder["client"] = FakeClient(options=options, **client_kwargs)
        return holder["client"]

    session = ClaudeSession(build_options("haiku"), client_factory=factory)
    return ClaudeAgentLLMService(model="haiku", session=session), holder


def context_with(text: str) -> LLMContext:
    return LLMContext(messages=[{"role": "user", "content": text}])


def test_options_are_isolated_and_toolless():
    opts = build_options("haiku")
    assert opts.model == "haiku"
    assert opts.tools == []
    assert opts.setting_sources == []
    assert opts.include_partial_messages is True
    assert opts.strict_mcp_config is True
    assert "Vera" in opts.system_prompt


def test_text_delta_and_last_user_text():
    assert text_delta(delta("Selam").event) == "Selam"
    assert text_delta({"type": "message_start"}) is None
    assert text_delta({"type": "content_block_delta", "delta": {"type": "input_json_delta"}}) is None
    ctx = LLMContext(
        messages=[
            {"role": "user", "content": "eski"},
            {"role": "assistant", "content": "cevap"},
            {"role": "user", "content": [{"type": "text", "text": "yeni"}, {"type": "text", "text": "soru"}]},
        ]
    )
    assert last_user_text(ctx) == "yeni soru"


async def test_stream_events_become_llm_text_frames():
    llm, holder = make_service(replies=[["Merhaba", " Mirza."]])
    down, _ = await run_test(
        llm,
        frames_to_send=[LLMContextFrame(context=context_with("Selam Vera")), SleepFrame(0.3)],
    )
    kinds = [type(f) for f in down]
    assert kinds.count(LLMFullResponseStartFrame) == 1
    assert kinds.count(LLMFullResponseEndFrame) == 1
    texts = [f.text for f in down if isinstance(f, LLMTextFrame)]
    assert texts == ["Merhaba", " Mirza."]
    assert kinds.index(LLMFullResponseStartFrame) < kinds.index(LLMTextFrame) < kinds.index(LLMFullResponseEndFrame)
    client = holder["client"]
    assert client.queries == ["Selam Vera"]
    assert client.connects == 1
    assert client.disconnected


async def test_one_client_for_many_turns_only_last_user_message_sent():
    llm, holder = make_service(replies=[["Bir."], ["İki."]])
    ctx = context_with("ilk")
    ctx2 = LLMContext(
        messages=[
            {"role": "user", "content": "ilk"},
            {"role": "assistant", "content": "Bir."},
            {"role": "user", "content": "ikinci"},
        ]
    )
    down, _ = await run_test(
        llm,
        frames_to_send=[LLMContextFrame(context=ctx), SleepFrame(0.3), LLMContextFrame(context=ctx2), SleepFrame(0.3)],
    )
    client = holder["client"]
    assert client.connects == 1
    assert client.queries == ["ilk", "ikinci"]
    assert [f.text for f in down if isinstance(f, LLMTextFrame)] == ["Bir.", "İki."]


async def test_interruption_calls_interrupt_drains_and_next_turn_is_clean():
    llm, holder = make_service(
        replies=[["Uzun ", "bir ", "cevap ", "bu ", "kesilecek."], ["Tamam, dinliyorum."]],
        delay=0.15,
    )
    down, _ = await run_test(
        llm,
        frames_to_send=[
            LLMContextFrame(context=context_with("anlat")),
            SleepFrame(0.4),  # ~2 deltas out
            InterruptionFrame(),
            SleepFrame(0.4),
            LLMContextFrame(context=context_with("dur")),
            SleepFrame(0.8),
        ],
    )
    client = holder["client"]
    assert client.interrupts == 1
    assert client.queries == ["anlat", "dur"]
    texts = [f.text for f in down if isinstance(f, LLMTextFrame)]
    assert "kesilecek." not in texts  # rest of the aborted turn was drained, not spoken
    assert texts[-1] == "Tamam, dinliyorum."
    assert len(texts) < 5 + 1
    assert llm.session.interrupts_sent == 1
