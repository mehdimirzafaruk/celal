"""CelalCore end to end with mocks: fake Hermes client, fake tunnel, fake TTS (no audio device)."""

import asyncio

import pytest
from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

import celal.pipeline as pipeline_mod
from celal.core import IDLE, SPEAKING, THINKING, CelalCore
from celal.hermes import Event
from celal.pipeline import Config
from celal.tunnel import OPEN


class FakeTTS(TTSService):
    def __init__(self, **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True, sample_rate=24000,
                         settings=TTSSettings(model="fake", voice="fake", language="tr"), **kwargs)
        self.spoken = []

    async def warmup(self):
        pass

    async def run_tts(self, text, context_id):
        self.spoken.append(text)
        yield TTSAudioRawFrame(audio=b"\x00\x00" * 2400, sample_rate=24000, num_channels=1, context_id=context_id)


class FakeTunnel:
    def __init__(self):
        self.status, self.detail, self._on_status = OPEN, "", None
        self.stopped = False
        self.kicks = 0

    def start(self):
        pass

    def kick(self, reason=""):
        self.kicks += 1
        return True

    async def stop(self):
        self.stopped = True


class FakeClient:
    def __init__(self, settings, replies=None, delay=0.0, online=True):
        self.settings = settings
        self.conversation = settings.conversation
        self.replies = replies or [["Merhaba Mirza. ", "Buradayım."]]
        self.delay = delay
        self.online = online
        self.prompts = []
        self.cancelled = 0

    async def ping(self):
        return (True, "bağlı") if self.online else (False, "sunucuya bağlanılamadı (ConnectError)")

    async def stream(self, text):
        self.prompts.append(text)
        parts = self.replies[min(len(self.prompts), len(self.replies)) - 1]
        try:
            for p in parts:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield Event("text", p)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

    def new_conversation(self):
        self.conversation = "yeni"
        return "yeni"

    async def aclose(self):
        pass


@pytest.fixture
def fake_tts(monkeypatch):
    tts = {}

    def make(cfg):
        tts["obj"] = FakeTTS()
        return tts["obj"]

    monkeypatch.setattr(pipeline_mod, "make_tts", make)
    return tts


async def make_core(settings, client, events):
    cfg = Config(play=False, print_sentences=False, printer=lambda s: None)
    core = CelalCore(settings, cfg, voice=False, emit=events.append, client=client, tunnel=FakeTunnel(),
                     health_interval=0.05)
    await core.start()
    return core


async def test_text_turn_is_answered_and_spoken(settings, fake_tts):
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    assert await core.wait_connected(2)
    assert await core.send_text("merhaba Celal")
    await asyncio.wait_for(core.turn_done.wait(), 10)
    kinds = [e["type"] for e in events]
    assert "user" in kinds and "assistant_start" in kinds and "assistant_end" in kinds
    end = [e for e in events if e["type"] == "assistant_end"][-1]
    assert end["text"] == "Merhaba Mirza. Buradayım."
    assert "Merhaba Mirza." in " ".join(fake_tts["obj"].spoken)
    states = [e["state"] for e in events if e["type"] == "status"]
    assert THINKING in states and SPEAKING in states and states[-1] == IDLE
    await core.shutdown()
    assert core.tunnel.stopped


async def test_disconnected_server_shows_bagli_degil_and_does_not_send(settings, fake_tts):
    events = []
    client = FakeClient(settings, online=False)
    core = await make_core(settings, client, events)
    await asyncio.sleep(0.2)
    snap = core.snapshot()
    assert snap["connected"] is False and snap["conn"] == "bağlı değil"
    assert await core.send_text("merhaba") is False
    assert client.prompts == []
    assert any(e["type"] == "error" and "bağlı değil" in e["text"] for e in events)
    # server comes back → connected again
    client.online = True
    await asyncio.sleep(0.2)
    assert core.snapshot()["conn"] == "bağlı"
    await core.shutdown()


async def test_durdur_cancels_inflight_request(settings, fake_tts):
    events = []
    client = FakeClient(settings, replies=[["bir "] + ["kelime "] * 50], delay=0.1)
    core = await make_core(settings, client, events)
    await core.send_text("uzun anlat")
    await asyncio.sleep(0.35)
    await core.interrupt()
    await asyncio.sleep(0.5)
    assert client.cancelled == 1
    end = [e for e in events if e["type"] == "assistant_end"]
    assert end and end[-1]["note"] == "kesildi"
    deltas_after = len([e for e in events if e["type"] == "assistant_delta"])
    await asyncio.sleep(0.3)
    assert len([e for e in events if e["type"] == "assistant_delta"]) == deltas_after  # stream really stopped
    assert core.state == IDLE
    await core.shutdown()


async def test_stalled_tunnel_is_kicked(settings, fake_tts):
    events = []
    client = FakeClient(settings, online=False)
    core = await make_core(settings, client, events)
    await asyncio.sleep(0.4)
    assert core.tunnel.kicks >= 1  # 3 failed pings while ssh claims to be open
    await core.shutdown()
