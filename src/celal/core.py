"""Celal's engine, shared by the window and the terminal mode.

Owns the SSH tunnel, the Hermes health check and one Pipecat pipeline
(optional mic → Whisper → Hermes → EMA/say → speaker). Everything runs on one
asyncio loop. The UI talks to it through ``send_text`` / ``set_mic`` /
``interrupt`` / ``new_conversation`` and receives plain dict events via
``emit``::

    {"type": "status", "state": "hazır|dinliyor|düşünüyor|konuşuyor",
     "connected": bool, "conn": "bağlı|bağlı değil", "reason": str,
     "tunnel": str, "mic": bool, "voice": bool}
    {"type": "user", "text": str}
    {"type": "assistant_start"} / {"type": "assistant_delta", "text": str} / {"type": "assistant_end"}
    {"type": "tool", "name": str}
    {"type": "error", "text": str}
    {"type": "info", "text": str}
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.workers.runner import WorkerRunner

from celal.config import Settings
from celal.hermes import HermesClient
from celal.llm_hermes import HermesLLMService
from celal.pipeline import STT_MODELS, Built, Config, build_text, build_voice
from celal.stt_filter import WhisperHallucinationFilter
from celal.tunnel import OPEN, TunnelManager

IDLE = "hazır"
LISTENING = "dinliyor"
THINKING = "düşünüyor"
SPEAKING = "konuşuyor"

CONNECTED = "bağlı"
DISCONNECTED = "bağlı değil"


class _FrameWatcher(BaseObserver):
    def __init__(self, core: CelalCore, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self.core = core

    async def on_push_frame(self, data: FramePushed):
        f = data.frame
        c = self.core
        if isinstance(f, BotStartedSpeakingFrame):
            c._bot_speaking = True
        elif isinstance(f, BotStoppedSpeakingFrame):
            c._bot_speaking = False
            c._last_audio = time.monotonic()
        elif isinstance(f, InterruptionFrame):
            if c._bot_speaking or c._llm_active:
                c.interruptions += 1
        elif isinstance(f, UserStartedSpeakingFrame):
            c._user_speaking = True
        elif isinstance(f, UserStoppedSpeakingFrame):
            c._user_speaking = False
        else:
            return
        c._refresh()


class _TranscriptWatcher(BaseObserver):
    """Reports what the user said, as it leaves the hallucination filter.

    Every push is observed so the filtered text is used, not Whisper's raw
    frame (which observers see first, before the echo gate and the filter).
    """

    def __init__(self, core: CelalCore, **kwargs):
        super().__init__(observe_every_push=True, **kwargs)
        self.core = core

    async def on_push_frame(self, data: FramePushed):
        f = data.frame
        if not isinstance(f, TranscriptionFrame) or not isinstance(data.source, WhisperHallucinationFilter):
            return
        c = self.core
        text = f.text.strip()
        if not text:
            return
        if c.built is not None:
            c.built.reply.reset()
            c.turn_done.clear()
        c._awaiting_reply = True
        c._awaiting_since = time.monotonic()
        c.emit({"type": "user", "text": text, "source": "mic"})
        c._refresh()


class CelalCore:
    def __init__(
        self,
        settings: Settings,
        cfg: Config,
        *,
        voice: bool,
        emit: Callable[[dict], None],
        client: HermesClient | None = None,
        tunnel: TunnelManager | None = None,
        health_interval: float = 3.0,
        stall_after: int = 3,
    ):
        self.settings = settings
        self.cfg = cfg
        self.voice = voice
        self._emit = emit
        self.client = client or HermesClient(settings)
        self.tunnel = tunnel or TunnelManager(settings)
        self.tunnel._on_status = self._on_tunnel
        self.health_interval = health_interval
        self.stall_after = stall_after
        self.connected = False
        self.reason = "başlıyor"
        self.mic = voice and cfg.mic_on_start
        self.state = IDLE
        self.built: Built | None = None
        self.llm: HermesLLMService | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []
        self._runner_task: asyncio.Task | None = None
        self._bot_speaking = False
        self._user_speaking = False
        self._llm_active = False
        self._awaiting_reply = False  # a prompt was sent, reply not yet fully spoken
        self._awaiting_since = 0.0
        self._last_audio = 0.0
        self._reply_text: list[str] = []
        self._last_status: dict | None = None
        self.interruptions = 0  # replies cut off (by the user, the Sus button or echo)
        self.turn_done = asyncio.Event()
        self.last_error: str | None = None

    # ---- events -------------------------------------------------------
    def emit(self, event: dict) -> None:
        try:
            self._emit(event)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"emit failed: {e}")

    def snapshot(self) -> dict:
        return {
            "type": "status",
            "state": self.state,
            "connected": self.connected,
            "conn": CONNECTED if self.connected else DISCONNECTED,
            "reason": self.reason,
            "tunnel": f"{self.tunnel.status}{' — ' + self.tunnel.detail if self.tunnel.detail else ''}",
            "mic": self.mic,
            "voice": self.voice,
            "conversation": self.client.conversation,
        }

    def _compute_state(self) -> str:
        window = self.built.window if self.built else None
        if self._bot_speaking or (window is not None and window.closed):
            return SPEAKING
        if self._llm_active or self._awaiting_reply:
            return THINKING
        if self.mic:
            return LISTENING
        return IDLE

    def _refresh(self) -> None:
        self.state = self._compute_state()
        snap = self.snapshot()
        if snap != self._last_status:
            self._last_status = snap
            self.emit(snap)

    def _on_tunnel(self, status: str, detail: str) -> None:
        self._refresh()

    def _on_llm(self, kind: str, value: str) -> None:
        if kind == "start":
            self._llm_active = True
            self._reply_text = []
            self.emit({"type": "assistant_start"})
        elif kind == "text":
            self._reply_text.append(value)
            self.emit({"type": "assistant_delta", "text": value})
        elif kind == "tool":
            self.emit({"type": "tool", "name": value})
        elif kind == "error":
            self.last_error = value
            self.emit({"type": "error", "text": value})
        elif kind == "end":
            self._llm_active = False
            self._last_audio = time.monotonic()
            self.emit({"type": "assistant_end", "text": "".join(self._reply_text), "note": value})
            if not self._reply_text:
                self._awaiting_reply = False
                self.turn_done.set()
        self._refresh()

    # ---- lifecycle ----------------------------------------------------
    async def start(self, warm: bool = True) -> None:
        self.loop = asyncio.get_running_loop()
        self.tunnel.start()
        self.llm = HermesLLMService(self.client, on_event=self._on_llm)
        cfg = self.cfg
        if not self.voice:
            self.built = build_text(cfg, self.llm, observers=[_FrameWatcher(self)])
        else:
            watchers = [_FrameWatcher(self), _TranscriptWatcher(self)]
            self.built = build_voice(cfg, self.llm, observers=watchers)
        started = asyncio.Event()

        @self.built.worker.event_handler("on_pipeline_started")
        async def _started(worker, frame):
            started.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.built.worker)
        self._runner_task = asyncio.create_task(runner.run(), name="pipeline")
        self._tasks.append(asyncio.create_task(self._health_loop(), name="health"))
        self._tasks.append(asyncio.create_task(self._tick(), name="state-tick"))
        await asyncio.wait_for(started.wait(), timeout=60)
        if warm:
            t = time.perf_counter()
            try:
                await self.built.tts.warmup()
            except Exception as e:  # noqa: BLE001
                self.emit({"type": "error", "text": f"TTS ısınamadı: {e}"})
            logger.info(f"TTS ready in {time.perf_counter() - t:.1f}s")
            if self.voice:
                self._tasks.append(asyncio.create_task(self._warm_whisper(), name="whisper-warm"))
        self._refresh()

    async def _warm_whisper(self) -> None:
        repo = STT_MODELS[self.cfg.stt]
        t = time.perf_counter()
        try:
            import mlx_whisper
            import numpy as np

            await asyncio.to_thread(
                mlx_whisper.transcribe, np.zeros(16000, dtype=np.float32), path_or_hf_repo=repo, language="tr"
            )
            self.emit({"type": "info", "text": f"Whisper hazır ({time.perf_counter() - t:.1f} s)"})
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "error", "text": f"Whisper yüklenemedi: {e}"})

    async def _health_loop(self) -> None:
        fails = 0
        while True:
            ok, reason = await self.client.ping()
            if ok != self.connected or reason != self.reason:
                if ok != self.connected:
                    logger.info(f"hermes: {'bağlı' if ok else 'bağlı değil'} ({reason})")
                self.connected, self.reason = ok, reason
                self._refresh()
            if ok:
                fails = 0
            elif self.tunnel.status == OPEN:
                fails += 1
                if fails >= self.stall_after:
                    self.tunnel.kick("(sağlık kontrolü cevapsız)")
                    fails = 0
            await asyncio.sleep(self.health_interval)

    async def _tick(self) -> None:
        """Settle 'düşünüyor' when a reply produced text but no audio (TTS failure)."""
        while True:
            await asyncio.sleep(0.3)
            if self._awaiting_reply and not self._llm_active and not self._bot_speaking:
                quiet = time.monotonic() - self._last_audio
                if quiet > 1.0 and self._reply_text_spoken_or_lost(quiet):
                    self._awaiting_reply = False
                    self.turn_done.set()
            self._refresh()

    def _reply_text_spoken_or_lost(self, quiet: float) -> bool:
        reply = self.built.reply if self.built else None
        if reply is not None and reply.done(settle=0.8):
            return True
        return quiet > 8.0

    async def wait_connected(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connected:
                return True
            ok, reason = await self.client.ping()
            if ok:
                self.connected, self.reason = True, reason
                self._refresh()
                return True
            await asyncio.sleep(0.5)
        return self.connected

    async def shutdown(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self.built is not None:
            try:
                await asyncio.wait_for(self.built.worker.cancel(reason="kapanış"), timeout=5)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"worker cancel: {e}")
        if self._runner_task is not None:
            try:
                await asyncio.wait_for(self._runner_task, timeout=5)
            except Exception:  # noqa: BLE001
                self._runner_task.cancel()
        await self.tunnel.stop()
        await self.client.aclose()

    # ---- actions ------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._llm_active or self._bot_speaking or self._awaiting_reply

    async def interrupt(self) -> None:
        if self.built is None:
            return
        await self.built.worker.queue_frame(InterruptionWorkerFrame(), FrameDirection.UPSTREAM)
        self._awaiting_reply = False
        self._refresh()

    async def send_text(self, text: str) -> bool:
        text = text.strip()
        if not text or self.built is None:
            return False
        if not self.connected:
            ok, reason = await self.client.ping()
            self.connected, self.reason = ok, reason
            if not ok:
                self._refresh()
                self.emit({"type": "error", "text": f"Sunucuya bağlı değil: {reason}. Mesaj gönderilmedi."})
                return False
        if self.busy:
            await self.interrupt()
            await asyncio.sleep(0.1)
        self.emit({"type": "user", "text": text, "source": "text"})
        self.last_error = None
        self.turn_done.clear()
        self.built.reply.reset()
        self._awaiting_reply = True
        self._awaiting_since = time.monotonic()
        self.built.timer.prompt_submitted()
        self._refresh()
        await self.built.worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True)
        )
        return True

    async def set_mic(self, on: bool) -> None:
        if not self.voice or self.built is None:
            self.emit({"type": "error", "text": "Mikrofon bu modda yok."})
            return
        inp = self.built.transport.input()
        setter = getattr(inp, "set_capturing", None)
        if setter is None:
            return
        try:
            setter(on)
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "error", "text": f"Mikrofon açılamadı: {e}"})
            return
        self.mic = on
        if on and self._bot_speaking:
            pass  # speaking continues until the user talks (VAD interrupts)
        self._refresh()

    async def new_conversation(self) -> None:
        if self.busy:
            await self.interrupt()
        name = self.client.new_conversation()
        self.emit({"type": "info", "text": f"Yeni sohbet: {name}"})
        self._refresh()

    # ---- thread bridge (UI threads) -----------------------------------
    def call(self, coro) -> None:
        """Schedule a coroutine on the core loop from another thread."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(coro, self.loop)
