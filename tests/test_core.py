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


# ---- turn ids (review finding: stale "kesildi" end finishing the new turn) ----


async def test_stale_interrupted_end_does_not_finish_new_turn(settings, fake_tts):
    """Old turn's 'end' arrives after the new turn began: the new turn stays open."""
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    core._begin_turn("eski soru")
    core._on_llm("start", "eski soru")
    t_new = core._begin_turn("yeni soru")  # what send_text does after the interruption
    core._on_llm("end", "kesildi")  # late, no reply text: the old code set turn_done here
    assert not core.turn_done.is_set() and core._awaiting_reply
    core._on_llm("start", "yeni soru")
    assert core._llm_turn == t_new
    core._on_llm("end", "")  # new stream, nothing to say → this one finishes the turn
    assert core.turn_done.is_set() and not core._awaiting_reply
    await core.shutdown()


async def test_stale_start_is_not_bound_to_new_turn(settings, fake_tts):
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    core._begin_turn("yeni")
    core._on_llm("start", "eski")  # a stream for an older prompt
    core._on_llm("end", "")
    assert not core.turn_done.is_set()
    await core.shutdown()


async def test_second_message_while_first_streams(settings, fake_tts):
    events = []
    client = FakeClient(settings, replies=[["bir "] * 40, ["İkinci cevap."]], delay=0.05)
    core = await make_core(settings, client, events)
    await core.send_text("uzun anlat")
    await asyncio.sleep(0.2)
    first = core.turn_id
    await core.send_text("kısa söyle")
    assert core.turn_id == first + 1
    await asyncio.wait_for(core.turn_done.wait(), 10)
    ends = [e for e in events if e["type"] == "assistant_end"]
    assert ends[-1]["text"] == "İkinci cevap."
    assert client.prompts == ["uzun anlat", "kısa söyle"]
    await core.shutdown()


# ---- health check backoff (review finding: restart every ~9 s forever) ----


async def test_hermes_down_backs_off_tunnel_restarts(settings, fake_tts):
    events = []
    client = FakeClient(settings, online=False)
    cfg = Config(play=False, print_sentences=False, printer=lambda s: None)
    core = CelalCore(settings, cfg, voice=False, emit=events.append, client=client, tunnel=FakeTunnel(),
                     health_interval=0.01, stall_after=3, max_kick_backoff=3)
    assert [core.stall_after * 2 ** min(k, 3) for k in range(5)] == [3, 6, 12, 24, 24]
    await core.start()
    await asyncio.sleep(1.0)  # ~100 failed pings
    kicks = core.tunnel.kicks
    # without backoff: ~33 restarts; with 3, 6, 12, 24, 24 … pings between them: about 6
    assert 2 <= kicks <= 8, kicks
    assert core.reason.startswith("Hermes yanıt vermiyor")
    client.online = True
    await asyncio.sleep(0.1)
    assert core.connected and core.kicks_since_ok == 0
    await core.shutdown()


# ---- Mac'te yap through the core ----


def _mac_runner(tmp_path, lines, **kw):
    import sys

    from celal.mac_task import MacTaskRunner
    from fakes_mac import FakeClaude, Spawner

    spawner = Spawner(lambda: FakeClaude(lines, **kw))
    return MacTaskRunner(claude=sys.executable, default_cwd=str(tmp_path), spawn=spawner), spawner


async def test_mac_prefix_runs_locally_and_speaks_summary(settings, fake_tts, tmp_path):
    from fakes_mac import init_line, result_line, tool_line

    events = []
    lines = [init_line(), tool_line("Bash", command="uv run pytest -q"), result_line("40 test geçti. Hata yok. Detay: …")]
    runner, spawner = _mac_runner(tmp_path, lines, delay=0.02)
    client = FakeClient(settings, online=False)  # the Mac path does not need Hermes
    cfg = Config(play=False, print_sentences=False, printer=lambda s: None)
    core = CelalCore(settings, cfg, voice=False, emit=events.append, client=client, tunnel=FakeTunnel(),
                     health_interval=0.05, mac_runner=runner)
    await core.start()
    (tmp_path / "proje").mkdir()
    assert await core.send_text(f"Mac'te {tmp_path}/proje içinde testleri çalıştır")
    task = core.last_mac_task
    assert task.cwd == str(tmp_path / "proje")
    assert await task.wait(5) and task.status == "done"
    await asyncio.wait_for(core.speech_done.wait(), 10)
    assert client.prompts == []  # Hermes never saw it
    mac = [e for e in events if e["type"] == "mac"]
    assert [e["phase"] for e in mac][0] == "start" and mac[-1]["phase"] == "done"
    assert any(e.get("text") == "komut çalıştırıyor: uv run pytest -q" for e in mac)
    try:
        assert mac[-1]["text"].startswith("40 test geçti"), mac[-1]
        assert mac[-1]["summary"] == "40 test geçti. Hata yok.", mac[-1]
        assert "40 test geçti. Hata yok." in " ".join(fake_tts["obj"].spoken), fake_tts["obj"].spoken
    finally:
        await core.shutdown()


async def test_mac_busy_toggle_and_durdur(settings, fake_tts, tmp_path):
    from fakes_mac import init_line

    events = []
    runner, spawner = _mac_runner(tmp_path, [init_line()], hold=True)
    client = FakeClient(settings)
    cfg = Config(play=False, print_sentences=False, printer=lambda s: None)
    core = CelalCore(settings, cfg, voice=False, emit=events.append, client=client, tunnel=FakeTunnel(),
                     health_interval=0.05, mac_runner=runner)
    await core.start()
    await core.wait_connected(2)
    await core.set_mac_mode(True)
    assert core.snapshot()["mac"]["mode"] is True
    assert await core.send_text("indirilenleri temizle")  # toggle on: no prefix needed
    assert core.snapshot()["mac"]["busy"] is True
    assert await core.send_text("Mac'te başka iş") is False
    assert any(e["type"] == "error" and e["text"] == "Mac'te bir iş zaten çalışıyor" for e in events)
    await core.interrupt()  # Durdur kills the claude process
    assert spawner.procs[0].terminated or spawner.procs[0].killed
    assert core.last_mac_task.status == "stopped" and not core.mac.busy
    await core.set_mac_mode(False)
    assert await core.send_text("merhaba")  # back to Hermes
    await asyncio.wait_for(core.turn_done.wait(), 10)
    assert client.prompts == ["merhaba"]
    await core.shutdown()
