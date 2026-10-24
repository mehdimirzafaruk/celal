"""SayTTSService with the real macOS `say` (fast, no download)."""

import shutil

import pytest
from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame
from pipecat.tests.utils import SleepFrame, run_test

from celal.tts_say import SayTTSService, say_to_pcm

pytestmark = pytest.mark.skipif(shutil.which("say") is None, reason="macOS `say` required")


async def test_say_to_pcm_returns_int16_mono():
    pcm = await say_to_pcm("Merhaba.", "Yelda", 24000)
    assert len(pcm) > 24000 * 2 * 0.2  # > 0.2 s of audio
    assert len(pcm) % 2 == 0


async def test_say_service_yields_audio_frames():
    spoken = []
    tts = SayTTSService(sample_rate=24000, on_sentence=spoken.append)
    down, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Merhaba Mirza."), SleepFrame(2.0)])
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert audio
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio)
    assert sum(len(f.audio) for f in audio) > 24000 * 2 * 0.5
    assert spoken == ["Merhaba Mirza."]
