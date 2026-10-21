"""EMALightningTTSService with a fake EMA engine (no model download)."""

import threading

import numpy as np
from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.tests.utils import SleepFrame, run_test

from jarvis.tts_ema import EMALightningTTSService, float_to_int16_bytes, iterate_in_thread


class FakeEMA:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []
        self.closed = threading.Event()

    def stream(self, text, speed=1.0, seed=None, sample_rate=48000):
        self.calls.append((text, speed, seed, sample_rate))

        def gen():
            try:
                yield from self.chunks
            finally:
                self.closed.set()

        return gen()


def test_float_to_int16_conversion_and_clipping():
    pcm = float_to_int16_bytes(np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0], dtype=np.float32))
    values = np.frombuffer(pcm, dtype="<i2").tolist()
    assert values == [0, 16383, -16383, 32767, -32767, 32767]


async def test_stream_chunks_become_tts_audio_frames():
    chunks = [np.full(2400, 0.25, dtype=np.float32), np.full(4800, -0.25, dtype=np.float32)]
    engine = FakeEMA(chunks)
    spoken = []
    tts = EMALightningTTSService(engine=engine, sample_rate=24000, on_sentence=spoken.append)
    down, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Merhaba Mirza."), SleepFrame(0.5)])
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert audio, f"no audio frames in {[type(f).__name__ for f in down]}"
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio)
    total = b"".join(f.audio for f in audio)
    assert len(total) == (2400 + 4800) * 2  # int16
    samples = np.frombuffer(total, dtype="<i2")
    assert samples[0] == int(0.25 * 32767) and samples[-1] == -int(0.25 * 32767)
    assert any(isinstance(f, TTSStartedFrame) for f in down)
    assert any(isinstance(f, TTSStoppedFrame) for f in down)
    assert engine.calls[0][0] == "Merhaba Mirza."
    assert engine.calls[0][3] == 24000
    assert spoken == ["Merhaba Mirza."]


async def test_early_stop_closes_ema_generator():
    engine = FakeEMA([np.zeros(10, dtype=np.float32)] * 100)
    agen = iterate_in_thread(lambda: engine.stream("x"))
    first = await agen.__anext__()
    assert len(first) == 10
    await agen.aclose()
    assert engine.closed.wait(2.0)
