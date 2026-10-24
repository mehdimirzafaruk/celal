"""HermesLLMService with a fake Hermes client."""

from pipecat.frames.frames import (
    ErrorFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

from celal.hermes import Event, HermesError
from celal.llm_hermes import HermesLLMService, last_user_text
from celal.speech_text import clean_for_speech


class FakeClient:
    def __init__(self, settings, events=None, error=None):
        self.settings = settings
        self.events = events or []
        self.error = error
        self.prompts = []

    async def stream(self, text):
        self.prompts.append(text)
        for e in self.events:
            yield e
        if self.error:
            raise self.error


def context(*messages):
    return LLMContext(messages=list(messages))


async def test_streams_text_between_start_and_end(settings):
    client = FakeClient(settings, [Event("tool", "web_search"), Event("text", "**Merhaba** "), Event("text", "Mirza 🙂.")])
    events = []
    llm = HermesLLMService(client, on_event=lambda k, v: events.append((k, v)))
    ctx = context({"role": "user", "content": "eski"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "merhaba"})
    down, _ = await run_test(llm, frames_to_send=[LLMContextFrame(context=ctx)])
    kinds = [type(f) for f in down if isinstance(f, (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame))]
    assert kinds[0] is LLMFullResponseStartFrame and kinds[-1] is LLMFullResponseEndFrame
    spoken = "".join(f.text for f in down if isinstance(f, LLMTextFrame))
    assert spoken == "Merhaba Mirza ."  # markdown/emoji removed for TTS
    assert client.prompts == ["merhaba"]  # only the newest user message
    assert ("tool", "web_search") in events
    assert [v for k, v in events if k == "text"] == ["**Merhaba** ", "Mirza 🙂."]  # UI gets raw text
    assert events[0] == ("start", "merhaba") and events[-1][0] == "end"


async def test_hermes_error_becomes_error_frame_and_event(settings):
    client = FakeClient(settings, error=HermesError("sunucuya bağlanılamadı (ConnectError)"))
    events = []
    llm = HermesLLMService(client, on_event=lambda k, v: events.append((k, v)))
    down, up = await run_test(llm, frames_to_send=[LLMContextFrame(context=context({"role": "user", "content": "x"}))])
    assert any(isinstance(f, ErrorFrame) and "bağlanılamadı" in f.error for f in up)
    assert ("error", "sunucuya bağlanılamadı (ConnectError)") in events
    assert isinstance(down[-1], LLMFullResponseEndFrame)


def test_last_user_text_handles_parts():
    ctx = context({"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
    assert last_user_text(ctx) == "a b"
    assert last_user_text(context({"role": "assistant", "content": "x"})) is None


def test_clean_for_speech():
    assert clean_for_speech("## Başlık\n- bir\n- iki `kod` 🚀") == " Başlık\nbir\niki kod "
    assert clean_for_speech("Saat 9'da.") == "Saat 9'da."
