"""macOS ``say -v Yelda`` as a Pipecat TTS service (fallback, no install).

``say`` cannot stream PCM to stdout, so each sentence is rendered to a
temporary WAV (16-bit, mono, at the pipeline rate) and then pushed as
``TTSAudioRawFrame`` chunks. TTFA is therefore the time to render one sentence.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import wave
from collections.abc import AsyncGenerator, Callable

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

CHUNK_SECONDS = 0.5


async def say_to_pcm(text: str, voice: str = "Yelda", sample_rate: int = 24000, rate: int | None = None) -> bytes:
    """Render ``text`` with macOS ``say`` and return mono int16 PCM at ``sample_rate``."""
    fd, path = tempfile.mkstemp(prefix="celal-say-", suffix=".wav")
    os.close(fd)
    try:
        cmd = ["say", "-v", voice, "-o", path, "--file-format=WAVE", f"--data-format=LEI16@{sample_rate}"]
        if rate:
            cmd += ["-r", str(rate)]
        cmd += ["--", text]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        try:
            _, err = await proc.communicate()
        except asyncio.CancelledError:
            proc.kill()
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"say failed ({proc.returncode}): {err.decode(errors='replace').strip()}")
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                raise RuntimeError("say produced unexpected WAV format")
            if w.getframerate() != sample_rate:
                raise RuntimeError(f"say produced {w.getframerate()} Hz, expected {sample_rate}")
            return w.readframes(w.getnframes())
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class SayTTSService(TTSService):
    """Turkish TTS through the built-in macOS Yelda voice."""

    def __init__(
        self,
        *,
        voice: str = "Yelda",
        rate: int | None = None,
        sample_rate: int = 24000,
        on_sentence: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=TTSSettings(model="macos-say", voice=voice, language="tr"),
            **kwargs,
        )
        self._voice = voice
        self._rate = rate
        self._say_rate = sample_rate
        self._on_sentence = on_sentence

    def can_generate_metrics(self) -> bool:
        return True

    async def warmup(self) -> None:
        await say_to_pcm("Merhaba.", self._voice, self._say_rate, self._rate)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = text.strip()
        if not text:
            return
        if self._on_sentence:
            self._on_sentence(text)
        try:
            await self.start_tts_usage_metrics(text)
            pcm = await say_to_pcm(text, self._voice, self._say_rate, self._rate)
            await self.stop_ttfb_metrics()
            step = int(self._say_rate * CHUNK_SECONDS) * 2
            for i in range(0, len(pcm), step):
                yield TTSAudioRawFrame(
                    audio=pcm[i : i + step],
                    sample_rate=self._say_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"say failed: {e}")
            yield ErrorFrame(error=f"say error: {e}")
        finally:
            await self.stop_ttfb_metrics()
