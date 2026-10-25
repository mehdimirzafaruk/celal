"""One Pipecat TTS service whose voice can be switched at runtime (see :mod:`celal.voices`).

Each engine yields mono int16 PCM at the pipeline rate. EMA Lightning and
macOS ``say`` run in this process; KittenTTS 2 and EMA-TTS run in a
:mod:`celal.voice_worker` process inside their own venv, one request at a time.
Worker processes start on first use and stop when another engine is chosen.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import subprocess
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from celal.tts_ema import EMALightningTTSService, float_to_int16_bytes, iterate_in_thread
from celal.tts_say import CHUNK_SECONDS, say_to_pcm
from celal.voices import BY_ID, DEFAULT_VOICE, VOICES, WORKER_PACKAGES, is_available, voices_dir, voices_env, worker_python


class EmaEngine:
    """EMA Lightning in this process."""

    def __init__(self, rate: int):
        self.rate = rate
        self._engine: Any | None = None

    def load(self) -> None:
        if self._engine is None:
            self._engine = EMALightningTTSService.load_engine("cpu")

    async def stream(self, text: str, speaker: str | None) -> AsyncIterator[bytes]:
        await asyncio.to_thread(self.load)
        engine = self._engine
        chunks = iterate_in_thread(lambda: engine.stream(text, speed=1.0, seed=7, sample_rate=self.rate))
        try:
            async for chunk in chunks:
                yield float_to_int16_bytes(chunk)
        finally:
            await chunks.aclose()

    def close(self) -> None:
        pass


class SayEngine:
    def __init__(self, rate: int, voice: str = "Yelda"):
        self.rate = rate
        self.voice = voice

    def load(self) -> None:
        pass

    async def stream(self, text: str, speaker: str | None) -> AsyncIterator[bytes]:
        pcm = await say_to_pcm(text, self.voice, self.rate)
        step = int(self.rate * CHUNK_SECONDS) * 2
        for i in range(0, len(pcm), step):
            yield pcm[i : i + step]

    def close(self) -> None:
        pass


class WorkerError(RuntimeError):
    pass


class WorkerEngine:
    """A :mod:`celal.voice_worker` process; one request at a time."""

    def __init__(self, worker: str, rate: int, normalize: Callable[[str], str] | None = None,
                 script: Path | None = None):
        self.worker = worker
        self.script = script or Path(__file__).with_name("voice_worker.py")
        self.rate = rate
        self.normalize = normalize
        self.info: dict = {}
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()  # serializes requests (and loading)

    def _read_frame(self) -> tuple[bytes, bytes]:
        assert self._proc is not None and self._proc.stdout is not None
        head = self._proc.stdout.read(5)
        if len(head) < 5:
            raise WorkerError(f"{self.worker} süreci kapandı (log: {self._log_path()})")
        kind, n = head[:1], struct.unpack("<I", head[1:])[0]
        payload = self._proc.stdout.read(n) if n else b""
        return kind, payload

    def _log_path(self) -> Path:
        return voices_dir() / "venvs" / f"{self.worker}.log"

    def load(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        python = worker_python(self.worker)
        if not python.exists():
            raise WorkerError(f"{self.worker} kurulu değil: celal --install-voice {self.worker}")
        log = open(self._log_path(), "ab")  # noqa: SIM115 - owned by the child
        logger.info(f"ses motoru başlıyor: {self.worker}")
        self._proc = subprocess.Popen(
            [str(python), "-u", str(self.script), self.worker],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env={**os.environ, **voices_env()},
        )
        log.close()
        kind, payload = self._read_frame()
        if kind != b"R":
            self._kill_locked()
            raise WorkerError(payload.decode(errors="replace") or f"{self.worker} açılamadı")
        self.info = json.loads(payload or b"{}")
        logger.info(f"ses motoru hazır: {self.worker} {self.info}")

    def _iter(self, text: str, speaker: str | None):
        """Blocking generator of PCM chunks; holds the lock until the reply ends."""
        with self._lock:
            self._start_locked()
            assert self._proc is not None and self._proc.stdin is not None
            req = {"text": self.normalize(text) if self.normalize else text, "voice": speaker, "rate": self.rate}
            self._proc.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode())
            self._proc.stdin.flush()
            done = False
            try:
                while True:
                    kind, payload = self._read_frame()
                    if kind == b"A":
                        yield payload
                    elif kind == b"E":
                        done = True
                        return
                    elif kind == b"X":
                        done = True
                        raise WorkerError(payload.decode(errors="replace"))
            finally:
                # Left early (interruption): read the rest so the next reply starts clean.
                while not done:
                    try:
                        kind, _ = self._read_frame()
                    except WorkerError:
                        break
                    done = kind in (b"E", b"X")

    async def stream(self, text: str, speaker: str | None) -> AsyncIterator[bytes]:
        chunks = iterate_in_thread(lambda: self._iter(text, speaker))
        try:
            async for chunk in chunks:
                yield chunk
        finally:
            await chunks.aclose()

    def _kill_locked(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            proc.kill()

    def close(self) -> None:
        # Not under the lock: a reply in flight ends with "process closed".
        self._kill_locked()


def turkish_normalizer() -> Callable[[str], str]:
    """Numbers, dates, money ... read aloud (EMA's normalizer-tr); Kitten's own is English."""
    try:
        from normalizer_tr import Normalizer
    except ImportError:
        return lambda text: text
    n = Normalizer()

    def normalize(text: str) -> str:
        try:
            return n.normalize(text, ambiguity_policy="fallback").normalized_text
        except Exception:  # noqa: BLE001
            return text

    return normalize



class SwitchableTTSService(TTSService):
    """Speaks each sentence with the selected voice; :meth:`select` switches between sentences."""

    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        sample_rate: int = 24000,
        on_sentence: Callable[[str], None] | None = None,
        engines: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=TTSSettings(model="celal-voices", voice=voice, language="tr"),
            **kwargs,
        )
        self._rate = sample_rate
        self._on_sentence = on_sentence
        self._engines: dict[str, Any] = dict(engines or {})
        if voice not in BY_ID or not is_available(BY_ID[voice]):
            voice = DEFAULT_VOICE
        self.voice_id = voice

    # engines are created on first use and the worker ones closed when left
    def _engine(self, name: str) -> Any:
        engine = self._engines.get(name)
        if engine is None:
            if name == "ema":
                engine = EmaEngine(self._rate)
            elif name == "say":
                engine = SayEngine(self._rate)
            else:
                engine = WorkerEngine(name, self._rate, turkish_normalizer() if name == "kitten" else None)
            self._engines[name] = engine
        return engine

    def voices(self) -> list[dict]:
        return [{"id": v.id, "label": v.label, "available": is_available(v)} for v in VOICES]

    async def select(self, voice_id: str) -> None:
        """Switch voice and load it (raises if it cannot load; the old voice stays)."""
        voice = BY_ID.get(voice_id)
        if voice is None:
            raise ValueError(f"bilinmeyen ses: {voice_id}")
        if not is_available(voice):
            hint = f" (celal --install-voice {voice.engine})" if voice.engine in WORKER_PACKAGES else ""
            raise WorkerError(f"{voice.label} kurulu değil{hint}")
        await asyncio.to_thread(self._engine(voice.engine).load)
        old = BY_ID[self.voice_id].engine
        self.voice_id = voice_id
        if old != voice.engine and old in WORKER_PACKAGES:
            self._engines.pop(old).close()  # free the big model's memory

    async def warmup(self) -> None:
        voice = BY_ID[self.voice_id]
        engine = self._engine(voice.engine)
        async for _ in engine.stream("Merhaba.", voice.speaker):
            pass

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = text.strip()
        if not text:
            return
        if self._on_sentence:
            self._on_sentence(text)
        voice = BY_ID[self.voice_id]
        try:
            await self.start_tts_usage_metrics(text)
            chunks = self._engine(voice.engine).stream(text, voice.speaker)
            try:
                async for audio in chunks:
                    await self.stop_ttfb_metrics()
                    if audio:
                        yield TTSAudioRawFrame(audio=audio, sample_rate=self._rate, num_channels=1,
                                               context_id=context_id)
            finally:
                await chunks.aclose()
        except Exception as e:  # noqa: BLE001
            logger.error(f"{voice.label} failed: {e}")
            yield ErrorFrame(error=f"{voice.label}: {e}")
        finally:
            await self.stop_ttfb_metrics()

    async def cleanup(self) -> None:
        await super().cleanup()
        for engine in self._engines.values():
            engine.close()
