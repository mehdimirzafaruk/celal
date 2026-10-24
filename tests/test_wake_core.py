"""Voice commands through the real voice pipeline (build_voice), the window status and the CLI entry point.

The mic is a short silent WAV (no audio device); Whisper MLX never runs because
no speech reaches it. Transcripts are injected after the input transport, so the
echo gate, hallucination filter, command gate, aggregator, turn sink and tab
router are all real; the Mac brain is a fake.
"""

import asyncio
import io
import sys
import wave

import pytest
from pipecat.frames.frames import TranscriptionFrame
from fakes_mac import FakeBrains, until
from test_core import FakeClient, FakeTTS, FakeTunnel

import celal.pipeline as pipeline_mod
from celal.core import IDLE, LISTENING, WAITING, CelalCore
from celal.pipeline import Config
from celal.wake import START_PHRASE, STOP_PHRASE


def silent_wav(path, secs=0.1, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * secs))
    return str(path)


@pytest.fixture
def voice_core(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "make_tts", lambda cfg: FakeTTS())

    async def make(client, events, wake=True):
        cfg = Config(play=False, print_sentences=False, printer=lambda s: None, mic_on_start=True, wake=wake,
                     input_wav=silent_wav(tmp_path / "silence.wav"), input_wav_tail_secs=0.2)
        brains = FakeBrains()
        core = CelalCore(settings, cfg, voice=True, emit=events.append, client=client, tunnel=FakeTunnel(),
                         health_interval=0.05, brain_factory=brains)
        await core.start(warm=False)  # warm=True would load Whisper
        core.brains = brains
        return core

    return make


async def hear(core, text):
    """What Whisper would push for a spoken sentence."""
    await core.built.worker.queue_frame(TranscriptionFrame(text=text, user_id="", timestamp="t"))


async def ask(core, text):
    """Speak while active and wait until the reply has been spoken (turn ends by the aggregator's stop timeout)."""
    n = len(core.brains.prompts)
    await hear(core, text)
    assert await until(lambda: len(core.brains.prompts) > n, 15)
    tab = core.tabs.active
    assert await until(lambda: not tab.busy, 15)
    await core.wait_quiet(10)
    await asyncio.sleep(0.5)  # echo tail after the spoken reply


def user_texts(events):
    return [e["line"]["text"] for e in events if e["type"] == "tab_line" and e["line"]["role"] == "user"]


def prompts(core):
    return [p for _, p in core.brains.prompts]


async def test_passive_until_start_phrase_then_active_until_stop_phrase(voice_core, settings):
    events = []
    client = FakeClient(settings, replies=[["Hava güzel."], ["İkinci."]])
    core = await voice_core(client, events)
    try:
        assert await core.wait_connected(2)
        assert core.mic and core.state == WAITING
        snap = core.snapshot()
        assert snap["wake"] == {"on": True, "active": False, "start": START_PHRASE, "stop": STOP_PHRASE}
        assert snap["state"] == WAITING

        # passive: ordinary speech, a Mac request and the stop phrase all vanish locally
        await hear(core, "hava nasıl")
        await hear(core, "Mac'te testleri çalıştır")
        await hear(core, "Celal dinlemeyi durdur")
        await asyncio.sleep(0.3)
        assert prompts(core) == [] and user_texts(events) == []
        assert core.built.listen.dropped == ["hava nasıl", "Mac'te testleri çalıştır", "Celal dinlemeyi durdur"]

        # the exact start phrase (as Whisper writes it) switches to active and is not a message
        await hear(core, "Celal, dinlemeye başla.")
        await asyncio.sleep(0.3)
        assert core.built.listen.active and core.state == LISTENING
        wake = [e for e in events if e["type"] == "wake"]
        assert wake and wake[-1]["active"] is True
        assert user_texts(events) == []

        # active: speech reaches the Mac tab (turn ends by the aggregator's stop timeout: no VAD here)
        await ask(core, "hava nasıl")
        assert prompts(core) == ["hava nasıl"]  # nothing from the passive phase was aggregated into it
        assert user_texts(events) == ["hava nasıl"]
        assert client.prompts == []  # Hermes is not the default brain any more
        assert core.state == LISTENING

        # the stop phrase returns to passive and is not a message
        await hear(core, "Celal dinlemeyi durdur")
        await asyncio.sleep(0.3)
        assert not core.built.listen.active and core.state == WAITING
        assert [e for e in events if e["type"] == "wake"][-1]["active"] is False
        await hear(core, "bunu duyma")
        await hear(core, "Celal dinlemeye başla")
        await ask(core, "ikinci soru")
        assert prompts(core) == ["hava nasıl", "ikinci soru"]
        assert user_texts(events) == ["hava nasıl", "ikinci soru"]
    finally:
        await core.shutdown()


async def test_no_wake_keeps_the_old_always_listening_behaviour(voice_core, settings):
    events = []
    client = FakeClient(settings)
    core = await voice_core(client, events, wake=False)
    try:
        assert core.state == LISTENING and core.snapshot()["wake"]["on"] is False
        await ask(core, "merhaba")
        assert prompts(core) == ["merhaba"] and user_texts(events) == ["merhaba"]
    finally:
        await core.shutdown()


async def test_mic_button_resets_to_passive_and_state_follows(voice_core, settings):
    events = []
    core = await voice_core(FakeClient(settings), events)
    try:
        core.built.listen.set_active(True)
        core._refresh()
        assert core.state == LISTENING
        await core.set_mic(False)
        assert core.state == IDLE and not core.built.listen.active
        await core.set_mic(True)
        assert core.state == WAITING
        assert core.snapshot()["wake"]["active"] is False
    finally:
        await core.shutdown()


# ---- window / menu bar ------------------------------------------------------


def test_menubar_title_shows_active_passive_and_off():
    from celal.ui import menubar_title

    assert menubar_title({"connected": True, "mic": False}) == "● Celal"
    assert menubar_title({"connected": False, "mic": False}) == "○ Celal"
    assert menubar_title({"connected": True, "mic": True, "wake": {"on": True, "active": False}}) == "● Celal ◌"
    assert menubar_title({"connected": True, "mic": True, "wake": {"on": True, "active": True}}) == "● Celal ⏺"
    assert menubar_title({"connected": True, "mic": True, "wake": {"on": False, "active": True}}) == "● Celal ⏺"


def test_page_knows_the_waiting_state_and_wake_events():
    from importlib import resources

    html = resources.files("celal").joinpath("web/index.html").read_text(encoding="utf-8")
    assert '"bekliyor"' in html and "waiting" in html
    assert 'case "wake"' in html


# ---- CLI entry point --------------------------------------------------------


class FakeInputStream:
    """sounddevice.RawInputStream stand-in: no device, never calls back."""

    def __init__(self, **kwargs):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        pass


def _cli_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CELAL_MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    monkeypatch.setenv("CELAL_CONFIG", str(tmp_path / "yok.toml"))
    monkeypatch.setenv("CELAL_MAC_API", "0")
    monkeypatch.setenv("CELAL_MAC_WARM", "0")  # no real Claude process in tests
    monkeypatch.setenv("CELAL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CELAL_PORT", "1")  # nothing answers: "bağlı değil" quickly
    monkeypatch.setenv("CELAL_API_KEY_FILE", str(tmp_path / "no-key"))
    monkeypatch.delenv("CELAL_SSH_HOST", raising=False)


def test_cli_no_wake_flag_reaches_the_pipeline_config(monkeypatch, tmp_path):
    from celal import cli

    _cli_env(monkeypatch, tmp_path)
    seen = {}

    async def fake_run_text(settings, cfg, args):
        seen["cfg"] = cfg
        return 0

    monkeypatch.setattr(cli, "_run_text", fake_run_text)
    assert cli.main(["--text", "--mic", "--tts", "say"]) == 0
    assert seen["cfg"].wake is True
    assert cli.main(["--text", "--mic", "--tts", "say", "--no-wake"]) == 0
    assert seen["cfg"].wake is False


def test_cli_text_mic_smoke_boots_passive(monkeypatch, tmp_path, capsys):
    """`celal --text --mic`: real entry point, real voice pipeline, fake mic/TTS, no server."""
    from celal import audio_io, cli
    from celal.core import CelalCore

    _cli_env(monkeypatch, tmp_path)
    monkeypatch.setattr(audio_io.sd, "RawInputStream", FakeInputStream)
    monkeypatch.setattr(pipeline_mod, "make_tts", lambda cfg: FakeTTS())

    async def no_whisper(self):
        pass

    monkeypatch.setattr(CelalCore, "_warm_whisper", no_whisper)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # EOF: exit right after boot
    code = cli.main(["--text", "--mic", "--tts", "say", "--no-play", "--connect-timeout", "0.2", "--log", "warning"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "dinleme pasif" in out and START_PHRASE in out, out
