"""EMA Lightning (Turkish TTS, CPU) as a streaming Pipecat TTS service.

``EMA.stream(text)`` is a blocking generator of float32 mono chunks in
[-1, 1] at the requested sample rate (first chunk ~1 s of audio, later ones
~4 s). It runs in a worker thread; chunks cross into asyncio through a queue
and become int16 ``TTSAudioRawFrame``s. Leaving the loop early (interruption)
closes the generator, which per EMA's docs drops the rest of that text's work.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable, Iterable, Iterator
from typing import Any

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

EMA_RATES = (48000, 24000, 16000, 8000)


def float_to_int16_bytes(chunk: Any) -> bytes:
    """float32 [-1, 1] (numpy or array-like) -> little-endian int16 PCM bytes."""
    audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


async def iterate_in_thread(make_iter: Callable[[], Iterable[Any]]) -> AsyncGenerator[Any, None]:
    """Drive a blocking iterator in a thread, yielding its items asynchronously.

    When the consumer stops early (``aclose``/cancellation) the thread is told
    to stop and the iterator is closed from inside that thread.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    done = object()

    def worker() -> None:
        it: Iterator[Any] | None = None
        try:
            it = iter(make_iter())
            for item in it:
                if stop.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as e:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
            loop.call_soon_threadsafe(queue.put_nowait, done)

    thread = threading.Thread(target=worker, name="ema-stream", daemon=True)
    thread.start()
    try:
        while True:
            item = await queue.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


class EMALightningTTSService(TTSService):
    """Local Turkish TTS with EMA Lightning (``pip install ema-lightning``)."""

    def __init__(
        self,
        *,
        engine: Any | None = None,
        device: str = "cpu",
        speed: float = 1.0,
        seed: int | None = 7,
        sample_rate: int = 24000,
        on_sentence: Callable[[str], None] | None = None,
        **kwargs,
    ):
        if sample_rate not in EMA_RATES:
            raise ValueError(f"EMA Lightning sample_rate must be one of {EMA_RATES}")
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=TTSSettings(model="canberkkkkkk/ema-lightning", voice="ema", language="tr"),
            **kwargs,
        )
        self._engine = engine
        self._device = device
        self._speed = speed
        self._seed = seed
        self._ema_rate = sample_rate
        self._on_sentence = on_sentence

    @staticmethod
    def load_engine(device: str = "cpu") -> Any:
        from ema_lightning import EMA

        return EMA(device=device)

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            logger.info("Loading EMA Lightning (first run downloads ~34 MB)...")
            self._engine = self.load_engine(self._device)
        return self._engine

    async def warmup(self) -> None:
        engine = await asyncio.to_thread(self._ensure_engine)
        await asyncio.to_thread(lambda: list(engine.stream("Merhaba.", sample_rate=self._ema_rate)))

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = text.strip()
        if not text:
            return
        if self._on_sentence:
            self._on_sentence(text)
        try:
            engine = self._ensure_engine()
            await self.start_tts_usage_metrics(text)
            kwargs: dict[str, Any] = {"speed": self._speed, "sample_rate": self._ema_rate}
            if self._seed is not None:
                kwargs["seed"] = self._seed
            chunks = iterate_in_thread(lambda: engine.stream(text, **kwargs))
            try:
                async for chunk in chunks:
                    await self.stop_ttfb_metrics()
                    audio = float_to_int16_bytes(chunk)
                    if not audio:
                        continue
                    yield TTSAudioRawFrame(
                        audio=audio,
                        sample_rate=self._ema_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
            finally:
                await chunks.aclose()
        except Exception as e:  # noqa: BLE001
            logger.error(f"EMA Lightning failed: {e}")
            yield ErrorFrame(error=f"EMA Lightning error: {e}")
        finally:
            await self.stop_ttfb_metrics()
