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
    {"type": "mac", "phase": "start|progress|done", ...}   # "Mac'te yap" task (mac_task.py)

Turns carry an id (``_turn``). A reply stream is bound to the turn whose
prompt it answers when it starts; events of an older, interrupted stream
(its late "kesildi" end) never finish the newer turn.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    TTSSpeakFrame,
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
from celal.mac_api import MacApiServer, ensure_token
from celal.mac_task import BUSY, MacBusy, MacTask, MacTaskRunner, parse_request, split_cwd
from celal.pipeline import STT_MODELS, Built, Config, build_text, build_voice
from celal.stt_filter import WhisperHallucinationFilter
from celal.tunnel import EXTERNAL, OPEN, ReverseTunnel, TunnelManager

IDLE = "hazır"
LISTENING = "dinliyor"
THINKING = "düşünüyor"
SPEAKING = "konuşuyor"

CONNECTED = "bağlı"
DISCONNECTED = "bağlı değil"
HERMES_DOWN = "Hermes yanıt vermiyor"


class _FrameWatcher(BaseObserver):
    def __init__(self, core: CelalCore, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self.core = core

    async def on_push_frame(self, data: FramePushed):
        f = data.frame
        c = self.core
        if isinstance(f, BotStartedSpeakingFrame):
            c._bot_speaking = True
            if c._speech_pending:
                c._speech_started = True
        elif isinstance(f, BotStoppedSpeakingFrame):
            c._bot_speaking = False
            c._last_audio = time.monotonic()
            if c._speech_pending and c._speech_started:
                c._speech_finished()
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
        if not text or c.mac_request(text) is not None:
            return  # spoken "Mac'te …": MacRouter hands it to the Mac task runner
        c._begin_turn(None)  # the aggregator may join transcripts: accept any reply start
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
        max_kick_backoff: int = 5,
        mac_runner: MacTaskRunner | None = None,
        mac_api: bool | None = None,
        reverse: ReverseTunnel | None = None,
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
        self.max_kick_backoff = max_kick_backoff
        self.kicks_since_ok = 0
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
        self._turn = 0  # id of the newest turn
        self._turn_prompt: str | None = None  # its typed prompt (None: mic / any)
        self._llm_turn: int | None = None  # turn the running reply stream answers
        # "Mac'te yap"
        self.mac = mac_runner or MacTaskRunner(
            claude=settings.mac_claude, default_cwd=settings.mac_cwd, timeout=settings.mac_timeout
        )
        self.mac._on_event = self._on_mac
        self.mac_mode = False  # window toggle: every message goes to the Mac
        self.mac_api_enabled = settings.mac_api if mac_api is None else mac_api
        self.mac_api: MacApiServer | None = None
        self.mac_api_status = "kapalı"
        self.reverse = reverse
        if self.reverse is None and self.mac_api_enabled:
            self.reverse = ReverseTunnel(settings)
        if self.reverse is not None:
            self.reverse._on_status = self._on_tunnel
        self.last_mac_task: MacTask | None = None
        self.speech_done = asyncio.Event()
        self.speech_done.set()
        self._speech_pending = False
        self._speech_started = False
        self._speech_queued = 0.0

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
            "mac": {
                "busy": self.mac.busy,
                "task": (self.mac.current.task[:80] if self.mac.current else ""),
                "mode": self.mac_mode,
                "api": self.mac_api_status,
                "reverse": (f"{self.reverse.status}" if self.reverse is not None else "kapalı"),
            },
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

    # ---- turns --------------------------------------------------------
    def _begin_turn(self, prompt: str | None) -> int:
        self._turn += 1
        self._turn_prompt = prompt
        self.turn_done.clear()
        if self.built is not None:
            self.built.reply.reset()
        self._awaiting_reply = True
        self._awaiting_since = time.monotonic()
        return self._turn

    def _finish_turn(self, turn: int | None = None) -> None:
        if turn is not None and turn != self._turn:
            return  # a stale stream (interrupted older turn) must not end the newer one
        self._awaiting_reply = False
        self.turn_done.set()

    @property
    def turn_id(self) -> int:
        return self._turn

    def _on_llm(self, kind: str, value: str) -> None:
        if kind == "start":
            if self._turn_prompt is None or value == self._turn_prompt:
                self._llm_turn = self._turn
            else:
                self._llm_turn = None
                logger.debug("reply stream for an older prompt started; it will not end the current turn")
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
            stream_turn, self._llm_turn = self._llm_turn, None
            self._llm_active = False
            self._last_audio = time.monotonic()
            self.emit({"type": "assistant_end", "text": "".join(self._reply_text), "note": value})
            if not self._reply_text and stream_turn is not None:
                self._finish_turn(stream_turn)
        self._refresh()

    # ---- Mac tasks ----------------------------------------------------
    def _on_mac(self, kind: str, task: MacTask) -> None:
        if kind == "start":
            self.emit({"type": "mac", "phase": "start", "task_id": task.task_id, "task": task.task,
                       "cwd": task.cwd, "source": task.source})
        elif kind == "progress":
            self.emit({"type": "mac", "phase": "progress", "task_id": task.task_id, "text": task.progress[-1]})
        elif kind == "done":
            self.emit({"type": "mac", "phase": "done", "task_id": task.task_id, "status": task.status,
                       "source": task.source, "text": task.result or task.error, "error": task.error,
                       "summary": task.summary, "seconds": task.seconds})
            if task.source != "hermes" and task.summary and self.loop is not None:
                # Hermes-started tasks answer through Hermes; only our own are spoken here.
                self._speech_begin()  # before the task runs: waiters must not see the old "done"
                self._tasks.append(asyncio.ensure_future(self.say(task.summary)))
        self._refresh()

    def mac_request(self, text: str, force: bool | None = None) -> tuple[str, str | None] | None:
        """(task, cwd) when the text should run on the Mac, else None."""
        parsed = parse_request(text, self.settings.mac_prefixes)
        if parsed is not None:
            return parsed if parsed[0] else None
        if force or (force is None and self.mac_mode):
            return split_cwd(text.strip())
        return None

    async def run_mac_task(
        self, task: str, cwd: str | None = None, *, shown: str | None = None, announce: bool = True
    ) -> MacTask | None:
        if announce:
            self.emit({"type": "user", "text": shown or task, "source": "text", "mac": True})
        try:
            t = self.mac.start(task, cwd, source="celal")
        except MacBusy:
            self.emit({"type": "error", "text": BUSY})
            return None
        except ValueError as e:
            self.emit({"type": "error", "text": f"Mac'te çalıştırılamadı: {e}"})
            return None
        self.last_mac_task = t
        return t

    def _route_spoken(self, text: str) -> bool:
        """MacRouter callback (pipeline task): True when the transcript is a Mac request."""
        req = self.mac_request(text)
        if req is None:
            return False
        self.emit({"type": "user", "text": text, "source": "mic", "mac": True})
        self._tasks.append(asyncio.ensure_future(self.run_mac_task(req[0], req[1], shown=None, announce=False)))
        return True

    async def set_mac_mode(self, on: bool) -> None:
        self.mac_mode = bool(on)
        self._refresh()

    async def say(self, text: str) -> None:
        """Speak a fixed text (Mac task summary) through the pipeline's TTS."""
        if self.built is None or not text.strip():
            self._speech_finished()
            return
        self._speech_begin()
        await self.built.worker.queue_frame(TTSSpeakFrame(text=text))

    def _speech_begin(self) -> None:
        self._speech_pending = True
        self._speech_started = self._bot_speaking  # queued behind a reply that is still playing
        self._speech_queued = time.monotonic()
        self.speech_done.clear()

    def _speech_finished(self) -> None:
        self._speech_pending = False
        self._speech_started = False
        self.speech_done.set()

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
            self.built = build_voice(cfg, self.llm, observers=watchers, mac_route=self._route_spoken)
        started = asyncio.Event()

        @self.built.worker.event_handler("on_pipeline_started")
        async def _started(worker, frame):
            started.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.built.worker)
        self._runner_task = asyncio.create_task(runner.run(), name="pipeline")
        self._tasks.append(asyncio.create_task(self._health_loop(), name="health"))
        if self.mac_api_enabled:
            await self._start_mac_api()
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

    async def _start_mac_api(self) -> None:
        try:
            token = ensure_token(self.settings.mac_api_key_file)
            self.mac_api = MacApiServer(self.mac, token, port=self.settings.mac_api_port)
            await self.mac_api.start()
            self.mac_api_status = f"127.0.0.1:{self.mac_api.port}"
        except OSError as e:
            self.mac_api = None
            self.mac_api_status = "açılamadı"
            self.emit({"type": "error", "text": f"Mac API açılamadı (127.0.0.1:{self.settings.mac_api_port}): {e.strerror or e}. "
                       "Hermes → Mac kapalı; başka bir Celal açık olabilir."})
            return
        if self.reverse is not None:
            self.reverse.start()

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

    def kick_threshold(self) -> int:
        """Failed pings before the next tunnel restart: 3, 6, 12, … (capped)."""
        return self.stall_after * (2 ** min(self.kicks_since_ok, self.max_kick_backoff))

    async def _health_loop(self) -> None:
        fails = 0
        while True:
            ok, reason = await self.client.ping()
            if not ok and self.tunnel.status in (OPEN, EXTERNAL):
                # ssh is up, the server's API is not answering: say so instead of
                # blaming the tunnel (restarting it does not help much).
                reason = f"{HERMES_DOWN}: {reason}"
            if ok != self.connected or reason != self.reason:
                if ok != self.connected:
                    logger.info(f"hermes: {'bağlı' if ok else 'bağlı değil'} ({reason})")
                self.connected, self.reason = ok, reason
                self._refresh()
            if ok:
                fails = 0
                self.kicks_since_ok = 0
            elif self.tunnel.status == OPEN:
                fails += 1
                if fails >= self.kick_threshold():
                    # Maybe the tunnel hangs; restart it, but back off when it does
                    # not help (Hermes itself down): 9 s, 18 s, 36 s … ~5 min apart.
                    self.tunnel.kick("(sağlık kontrolü cevapsız)")
                    self.kicks_since_ok += 1
                    fails = 0
            await asyncio.sleep(self.health_interval)

    async def _tick(self) -> None:
        """Settle 'düşünüyor' when a reply produced text but no audio (TTS failure)."""
        while True:
            await asyncio.sleep(0.3)
            now = time.monotonic()
            if self._awaiting_reply and not self._llm_active and not self._bot_speaking:
                quiet = now - self._last_audio
                if quiet > 1.0 and now - self._awaiting_since > 1.0 and self._reply_text_spoken_or_lost(quiet):
                    self._finish_turn()
            if self._speech_pending and not self._bot_speaking and now - self._speech_queued > 15.0:
                self._speech_finished()  # TTS produced nothing (error) or no audio events
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
        try:
            try:
                await self.mac.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"mac shutdown: {e}")
            if self.mac_api is not None:
                await self.mac_api.stop()
            if self.reverse is not None:
                await self.reverse.stop()
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
        finally:
            await self.tunnel.stop()
            await self.client.aclose()

    # ---- actions ------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._llm_active or self._bot_speaking or self._awaiting_reply

    async def _interrupt_speech(self) -> None:
        if self.built is None:
            return
        await self.built.worker.queue_frame(InterruptionWorkerFrame(), FrameDirection.UPSTREAM)
        self._awaiting_reply = False
        self._refresh()

    async def interrupt(self) -> None:
        """Durdur: stop speaking, cancel the Hermes request, kill a running Mac task."""
        await self._interrupt_speech()
        if self.mac.busy:
            await self.mac.stop()
            self._refresh()

    async def send_text(self, text: str, mac: bool | None = None) -> bool:
        """Send a typed message to Hermes, or run it on the Mac ("Mac'te …" / toggle / ``mac=True``)."""
        text = text.strip()
        if not text or self.built is None:
            return False
        req = self.mac_request(text, mac)
        if req is not None:
            return await self.run_mac_task(req[0], req[1], shown=text) is not None
        if not self.connected:
            ok, reason = await self.client.ping()
            self.connected, self.reason = ok, reason
            if not ok:
                self._refresh()
                self.emit({"type": "error", "text": f"Sunucuya bağlı değil: {reason}. Mesaj gönderilmedi."})
                return False
        if self.busy:
            await self._interrupt_speech()
            await asyncio.sleep(0.1)  # let the interruption reach the LLM before the new prompt
        self.emit({"type": "user", "text": text, "source": "text"})
        self.last_error = None
        self._begin_turn(text)
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
            await self._interrupt_speech()
        name = self.client.new_conversation()
        self.emit({"type": "info", "text": f"Yeni sohbet: {name}"})
        self._refresh()

    # ---- thread bridge (UI threads) -----------------------------------
    def call(self, coro) -> None:
        """Schedule a coroutine on the core loop from another thread."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(coro, self.loop)
