"""CelalCore end to end with mocks: fake brains (Mac Claude), fake Hermes client, fake tunnel, fake TTS."""

import asyncio

import pytest
from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

import celal.pipeline as pipeline_mod
from celal.brains import HermesBrain
from celal.core import IDLE, SPEAKING, THINKING, CelalCore
from celal.hermes import Event
from celal.pipeline import Config
from celal.tunnel import OPEN
from fakes_mac import FakeBrains, until


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


class Brains(FakeBrains):
    """Mac tabs: fake Claude; Hermes tabs: the real HermesBrain on a fake Hermes client."""

    def __init__(self, client, core_ref, **kw):
        super().__init__(**kw)
        self.client = client
        self.core_ref = core_ref

    def __call__(self, tab):
        if tab.kind == "hermes":
            return HermesBrain(self.client, ensure_connected=lambda: self.core_ref[0]._ensure_connected())
        return super().__call__(tab)


async def make_core(settings, client, events, brains=None, **kw):
    cfg = Config(play=False, print_sentences=False, printer=lambda s: None)
    ref = []
    brains = brains or Brains(client, ref)
    brains.client, brains.core_ref = client, ref
    core = CelalCore(settings, cfg, voice=False, emit=events.append, client=client, tunnel=FakeTunnel(),
                     health_interval=0.05, brain_factory=brains, **kw)
    ref.append(core)
    await core.start()
    core.brains = brains
    return core


def spoken(fake_tts):
    return " ".join(fake_tts["obj"].spoken)


async def answered(core, tab, n=1, timeout=5):
    assert await until(lambda: tab.finished >= n, timeout)
    await core.wait_quiet(10)


async def test_message_goes_to_the_mac_and_the_answer_is_spoken(settings, fake_tts):
    events = []
    client = FakeClient(settings)
    core = await make_core(settings, client, events)
    tab = await core.send_text("bu klasörde kaç dosya var")
    assert tab.kind == "mac" and tab.id == core.tabs.active_id
    await answered(core, tab)
    assert core.brains.prompts == [("mac", "bu klasörde kaç dosya var")]
    assert client.prompts == []  # Hermes was not asked
    assert "Tamam." in spoken(fake_tts) and "bu klasörde kaç dosya var bitti." in spoken(fake_tts)
    lines = [e["line"]["role"] for e in events if e["type"] == "tab_line"]
    assert lines == ["user", "progress", "assistant"]
    states = [e["state"] for e in events if e["type"] == "status"]
    assert THINKING in states and SPEAKING in states and states[-1] == IDLE
    timing = [e for e in events if e["type"] == "timing"]
    assert timing and "ilk ses" in timing[-1]["text"]
    assert core.timings[-1]["first_audio"] is not None
    await core.shutdown()
    assert core.tunnel.stopped


async def test_follow_up_goes_to_the_same_tab_and_client(settings, fake_tts):
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    tab = await core.send_text("ilk soru")
    await answered(core, tab)
    again = await core.send_text("takip sorusu")
    assert again is tab
    await answered(core, tab, 2)
    assert len(core.brains.created) == 1
    await core.shutdown()


async def test_sunucuda_prefix_goes_to_hermes_and_mac_stays_default(settings, fake_tts):
    events = []
    client = FakeClient(settings, replies=[["Sunucu iyi."]])
    core = await make_core(settings, client, events)
    first = core.tabs.active
    htab = await core.send_text("sunucuda disk durumu ne")
    assert htab.kind == "hermes" and htab.label.startswith("Hermes:") and core.tabs.active_id == htab.id
    await answered(core, htab)
    assert client.prompts == ["sunucuda disk durumu ne"]
    assert "Sunucu iyi." in spoken(fake_tts)
    back = await core.send_text("peki bu klasörde ne var")  # no prefix: Mac first, even from a Hermes tab
    assert back is first and back.kind == "mac" and core.tabs.active_id == first.id
    await answered(core, back)
    assert client.prompts == ["sunucuda disk durumu ne"]
    await core.shutdown()


async def test_yeni_gorev_opens_a_new_tab(settings, fake_tts):
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    first = await core.send_text("birinci iş")
    await answered(core, first)
    second = await core.send_text("yeni görev README'yi güncelle")
    assert second is not first and second.label == "README'yi güncelle"
    assert core.tabs.active_id == second.id
    await answered(core, second)
    assert ("mac", "README'yi güncelle") in core.brains.prompts
    await core.shutdown()


async def test_background_tab_announces_completion_in_one_sentence(settings, fake_tts):
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    bg = await core.send_text("readme görevi bekle")
    assert await until(lambda: bg.status == "running")
    fg = await core.new_tab()
    assert core.tabs.active_id == fg.id
    core.brains.release.set()
    assert await until(lambda: bg.finished == 1)
    await core.wait_quiet(10)
    said = fake_tts["obj"].spoken
    assert "readme görevi bekle görevi bitti." in said
    assert not any("Tamam." in s for s in said)  # its full answer is not read aloud
    assert core.tabs.get(bg.id).unread
    await core.shutdown()


async def test_durdur_stops_the_active_mac_task_with_visible_feedback(settings, fake_tts):
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    tab = await core.send_text("uzun iş bekle")
    assert await until(lambda: tab.status == "running")
    res = await core.interrupt()
    assert res["stopped"] is True and core.brains.created[0].interrupts == 1
    kinds = [e["type"] for e in events]
    assert kinds.index("stopping") < kinds.index("stopped")  # immediate feedback first
    stopped = [e for e in events if e["type"] == "stopped"][-1]
    assert stopped["tab_id"] == tab.id and "Mac görevi durduruldu" in stopped["text"]
    assert tab.status == "stopped" and core.state == IDLE
    await core.shutdown()


async def test_durdur_never_cancels_hermes_work(settings, fake_tts):
    events = []
    client = FakeClient(settings, replies=[["bir "] + ["kelime "] * 30], delay=0.05)
    core = await make_core(settings, client, events)
    tab = await core.send_text("sunucuda uzun anlat")
    assert await until(lambda: client.prompts)
    await asyncio.sleep(0.2)
    res = await core.interrupt()
    assert res["server"] is True and res["stopped"] is False
    stopped = [e for e in events if e["type"] == "stopped"][-1]
    assert "sunucudaki iş sürüyor" in stopped["text"]
    assert await until(lambda: tab.finished == 1, 5)
    assert client.cancelled == 0 and tab.status == "done"  # the stream was not cancelled
    await core.shutdown()


async def test_durdur_cuts_the_speech(settings, fake_tts):
    events = []
    brains = Brains(None, [], replies={"anlat": [f"Cümle {i}. " for i in range(12)]}, delay=0.05)
    core = await make_core(settings, FakeClient(settings), events, brains=brains)
    tab = await core.send_text("uzun anlat")
    assert await until(lambda: core._bot_speaking, 5)
    res = await core.interrupt()
    assert res["speech"] is True
    await answered(core, tab)
    assert len(fake_tts["obj"].spoken) < 12  # the rest of the answer was not read
    assert any(e["type"] == "stopped" and "ses kesildi" in e["text"] for e in events)
    await core.shutdown()


async def test_durdur_with_nothing_running_says_so(settings, fake_tts):
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    res = await core.interrupt()
    assert res["text"] == "Durduracak bir şey yoktu."
    await core.shutdown()


async def test_hermes_tab_without_server_shows_error(settings, fake_tts):
    events = []
    client = FakeClient(settings, online=False)
    core = await make_core(settings, client, events)
    await asyncio.sleep(0.2)
    assert core.snapshot()["conn"] == "bağlı değil"
    tab = await core.send_text("sunucuda merhaba")
    assert await until(lambda: tab.finished == 1)
    assert client.prompts == [] and tab.status == "error"
    assert "Sunucuya bağlı değil" in tab.lines[-1]["text"]
    # the Mac still answers without the server
    mac = await core.send_text("merhaba")
    await answered(core, mac)
    assert mac.status == "done"
    client.online = True
    await asyncio.sleep(0.2)
    assert core.snapshot()["conn"] == "bağlı"
    await core.shutdown()


async def test_snapshot_lists_tabs_and_history_for_the_page(settings, fake_tts):
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    tab = await core.send_text("selam")
    await answered(core, tab)
    snap = core.snapshot()
    assert snap["active"] == tab.id and snap["tabs"][0]["label"] == "selam"
    assert snap["mac"]["max"] == 3
    hist = core.history()
    assert hist[0]["tab_id"] == tab.id and hist[0]["lines"][0]["text"] == "selam"
    await core.shutdown()


async def test_stalled_tunnel_is_kicked(settings, fake_tts):
    events = []
    client = FakeClient(settings, online=False)
    core = await make_core(settings, client, events)
    await asyncio.sleep(0.4)
    assert core.tunnel.kicks >= 1  # 3 failed pings while ssh claims to be open
    await core.shutdown()


# ---- health check backoff (review finding: restart every ~9 s forever) ----


async def test_hermes_down_backs_off_tunnel_restarts(settings, fake_tts):
    events = []
    client = FakeClient(settings, online=False)
    cfg = Config(play=False, print_sentences=False, printer=lambda s: None)
    core = CelalCore(settings, cfg, voice=False, emit=events.append, client=client, tunnel=FakeTunnel(),
                     health_interval=0.01, stall_after=3, max_kick_backoff=3, brain_factory=FakeBrains())
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


class VoiceTTS(FakeTTS):
    """FakeTTS with the switchable-voice surface the window uses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voice_id = "ema"

    def voices(self):
        return [{"id": "ema", "label": "EMA Lightning", "available": True},
                {"id": "kitten-haber", "label": "Kitten · Haber spikeri", "available": True}]

    async def select(self, voice_id):
        if voice_id == "bozuk":
            raise RuntimeError("model yok")
        self.voice_id = voice_id


async def test_set_voice_switches_saves_and_speaks(settings, monkeypatch):
    from celal.voices import load_choice

    tts = {}
    monkeypatch.setattr(pipeline_mod, "make_tts", lambda cfg: tts.setdefault("obj", VoiceTTS()))
    events = []
    core = await make_core(settings, FakeClient(settings), events)
    assert core.snapshot()["tts"]["id"] == "ema"
    await core.set_voice("kitten-haber")
    assert core.snapshot()["tts"] == {"id": "kitten-haber", "voices": tts["obj"].voices(), "loading": None}
    assert load_choice(settings.state_dir) == "kitten-haber"
    assert any(e.get("type") == "info" and "Haber spikeri" in e["text"] for e in events)
    assert await until(lambda: any("bu sesle" in s for s in tts["obj"].spoken), 3)
    await core.set_voice("bozuk")
    assert tts["obj"].voice_id == "kitten-haber"
    assert any(e.get("type") == "error" and "model yok" in e["text"] for e in events)
    assert load_choice(settings.state_dir) == "kitten-haber"
    await core.shutdown()
