"""Switchable voices: registry, remembered choice, worker protocol, runtime switching."""

import os
import sys
import textwrap

import numpy as np
import pytest
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.tests.utils import SleepFrame, run_test

from celal import voice_worker
from celal.tts_switch import SwitchableTTSService, WorkerEngine, WorkerError, turkish_normalizer
from celal.voices import BY_ID, VOICES, is_available, load_choice, save_choice, voices_dir, voices_env, worker_python

# A worker that speaks the protocol without a model: N samples of value v per request.
FAKE_WORKER = textwrap.dedent('''
    import json, struct, sys
    out = sys.stdout.buffer
    def frame(k, p=b""):
        out.write(k + struct.pack("<I", len(p)) + p); out.flush()
    if sys.argv[1] == "broken":
        frame(b"X", "model yok".encode()); sys.exit(1)
    frame(b"R", json.dumps({"rate": 24000}).encode())
    for line in sys.stdin:
        req = json.loads(line)
        if req["text"] == "hata":
            frame(b"X", b"bozuk"); continue
        voice = req.get("voice") or ""
        for _ in range(3):
            frame(b"A", struct.pack("<h", len(voice)) * 100)
        frame(b"E")
''')


class FakeEngine:
    def __init__(self, name):
        self.name = name
        self.loaded = 0
        self.closed = 0
        self.calls = []

    def load(self):
        self.loaded += 1

    async def stream(self, text, speaker):
        self.calls.append((text, speaker))
        yield b"\x01\x00" * 240

    def close(self):
        self.closed += 1


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setenv("CELAL_VOICES_DIR", str(tmp_path))
    return tmp_path


def fake_venv(models, name):
    py = worker_python(name)
    py.parent.mkdir(parents=True)
    py.symlink_to(sys.executable)
    return py


def test_registry_ids_unique_and_default_present():
    assert len({v.id for v in VOICES}) == len(VOICES)
    assert {"ema", "ema-tts", "say"} <= set(BY_ID)
    assert {v.speaker for v in VOICES if v.engine == "kitten"} == {
        "TurkishNewsAnchor", "TurkishFisherman", "TurkishGrandmaCooking"}


def test_availability_follows_venv(models):
    assert is_available(BY_ID["say"]) and is_available(BY_ID["ema"])
    assert not is_available(BY_ID["kitten-haber"])
    fake_venv(models, "kitten")
    assert is_available(BY_ID["kitten-balikci"])


def test_choice_roundtrip_and_unknown_ignored(tmp_path):
    assert load_choice(tmp_path) is None
    save_choice(tmp_path, "kitten-haber")
    assert load_choice(tmp_path) == "kitten-haber"
    (tmp_path / "voice.json").write_text('{"voice": "yok-boyle"}')
    assert load_choice(tmp_path) is None


def test_unavailable_start_voice_falls_back(models):
    tts = SwitchableTTSService(voice="kitten-haber")
    assert tts.voice_id == "ema"


def test_voices_live_on_the_mac_disk(models, monkeypatch):
    monkeypatch.delenv("CELAL_VOICES_DIR")
    assert str(voices_dir()).startswith(os.path.expanduser("~/"))
    assert "KIOXIA" not in str(worker_python("kitten"))
    env = voices_env({"CELAL_VOICES_DIR": "/tmp/v"})
    assert env["HF_HUB_CACHE"] == "/tmp/v/huggingface/hub" and env["XDG_CACHE_HOME"] == "/tmp/v/cache"


async def test_select_switches_and_closes_left_worker(models):
    fake_venv(models, "kitten")
    fake_venv(models, "ema-tts")
    engines = {n: FakeEngine(n) for n in ("ema", "kitten", "ema-tts")}
    tts = SwitchableTTSService(voice="ema", engines=engines)
    await tts.select("kitten-babaanne")
    assert tts.voice_id == "kitten-babaanne" and engines["kitten"].loaded == 1
    await tts.select("kitten-haber")  # same worker, other speaker: stays open
    assert engines["kitten"].closed == 0
    await tts.select("ema-tts")
    assert engines["kitten"].closed == 1
    with pytest.raises(ValueError):
        await tts.select("yok")


async def test_select_uninstalled_keeps_old_voice(models):
    tts = SwitchableTTSService(voice="ema", engines={"ema": FakeEngine("ema")})
    with pytest.raises(WorkerError, match="install-voice kitten"):
        await tts.select("kitten-haber")
    assert tts.voice_id == "ema"


async def test_sentences_use_the_selected_speaker(models):
    fake_venv(models, "kitten")
    engines = {"ema": FakeEngine("ema"), "kitten": FakeEngine("kitten")}
    tts = SwitchableTTSService(voice="ema", engines=engines)
    await tts.select("kitten-balikci")
    down, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Selam."), SleepFrame(0.3)])
    assert any(isinstance(f, TTSAudioRawFrame) for f in down)
    assert engines["kitten"].calls == [("Selam.", "TurkishFisherman")]
    assert engines["ema"].calls == []


async def test_worker_engine_protocol(models, tmp_path):
    fake_venv(models, "kitten")
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    engine = WorkerEngine("kitten", 24000, normalize=str.lower, script=script)
    try:
        engine.load()
        assert engine.info == {"rate": 24000}
        chunks = [c async for c in engine.stream("merhaba", "Abc")]
        assert len(chunks) == 3 and all(len(c) == 200 for c in chunks)
        assert np.frombuffer(chunks[0], "<i2")[0] == 3
        with pytest.raises(WorkerError, match="bozuk"):
            async for _ in engine.stream("HATA", None):  # normalized to "hata"
                pass
        # Leaving a reply early drains it: the next one starts clean.
        agen = engine.stream("bir", "X")
        await agen.__anext__()
        await agen.aclose()
        again = [c async for c in engine.stream("iki", "XY")]
        assert len(again) == 3 and np.frombuffer(again[0], "<i2")[0] == 2
    finally:
        engine.close()


async def test_worker_load_error_is_reported(models, tmp_path):
    fake_venv(models, "broken")
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    engine = WorkerEngine("broken", 24000, script=script)
    with pytest.raises(WorkerError, match="model yok"):
        engine.load()


async def test_worker_error_becomes_error_frame(models, tmp_path):
    fake_venv(models, "ema-tts")
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    worker = WorkerEngine("ema-tts", 24000, script=script)
    tts = SwitchableTTSService(voice="ema-tts", engines={"ema-tts": worker})
    try:
        down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="hata"), SleepFrame(0.5)])
    finally:
        worker.close()
    assert any(isinstance(f, ErrorFrame) and "bozuk" in f.error for f in down + up)


def test_turkish_normalizer_reads_numbers():
    assert turkish_normalizer()("%15 indirim") == "yüzde on beş indirim"


def test_worker_resamples_to_requested_rate():
    pcm = voice_worker._to_int16(np.zeros(48000, dtype=np.float32), 48000, 24000)
    assert len(pcm) == 24000 * 2
    assert os.path.basename(voice_worker.__file__) == "voice_worker.py"
