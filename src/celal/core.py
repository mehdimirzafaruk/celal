"""Celal's engine, shared by the window and the terminal mode.

Owns the SSH tunnel to Hermes (health checked), the tabs (:mod:`celal.tabs`:
one Mac Claude Code or Hermes conversation each, up to ``max_parallel``
answering at once), the Mac API for Hermes and one Pipecat pipeline that is
only ears and mouth (mic → Whisper → turn → router; text → TTS → speaker).

Routing (:func:`celal.mac_task.route`): every message goes to the **active
tab**, which is the Mac's Claude Code unless it is a Hermes tab. "sunucuda …"
/ "Hermes'e sor …" opens (or uses) a Hermes tab, "yeni görev …" a new tab.

Speech: only the active tab's reply is spoken, sentence by sentence as it
streams. A background tab that finishes says one short sentence ("README
görevi bitti") and gets an unread dot. Durdur (:meth:`interrupt`) cuts the
speech and stops the active tab's Mac task; work sent to Hermes is never
cancelled from here.

Events for the UI (plain dicts through ``emit``)::

    {"type": "status", "state": "hazır|bekliyor|dinliyor|düşünüyor|konuşuyor", "connected": bool,
     "conn": str, "reason": str, "tunnel": str, "mic": bool, "voice": bool, "wake": {...},
     "tabs": [tab view...], "active": tab id, "mac": {"running", "queued", "max", "api", "reverse"}}
    {"type": "tab_line", "tab_id": str, "line": {"role": "user|assistant|progress|info|error", "text": str, ...}}
    {"type": "tab_delta", "tab_id": str, "text": str} / {"type": "tab_break", "tab_id": str}
    {"type": "tab_turn", "tab_id": str, "phase": "start|end", "status": str, ...}
    {"type": "stopping", "tab_id": str} / {"type": "stopped", "tab_id": str, "text": str}
    {"type": "wake", "active": bool, "text": str}
    {"type": "timing", "tab_id": str, "text": str}
    {"type": "error", "text": str} / {"type": "info", "text": str}
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
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.workers.runner import WorkerRunner

from celal.brains import TurnOutcome, make_brain_factory, scrub_parent_env
from celal.config import Settings
from celal.hermes import HermesClient
from celal.mac_api import MacApiServer, ensure_token
from celal.mac_task import HERMES, MAC_FORCED, NEW, route, same_dir, spoken_summary, split_cwd
from celal.pipeline import STT_MODELS, Built, Config, build_text, build_voice
from celal.speech_text import SentenceBuffer
from celal.voices import BY_ID, save_choice
from celal.tabs import MAC, QUEUED, RUNNING, Tab, TabManager, make_label
from celal.tunnel import EXTERNAL, OPEN, ReverseTunnel, TunnelManager
from celal.wake import START_PHRASE, STOP_PHRASE, ListenState

IDLE = "hazır"
WAITING = "bekliyor"  # mic open, passive: only "Celal dinlemeye başla" is heard
LISTENING = "dinliyor"
THINKING = "düşünüyor"
SPEAKING = "konuşuyor"

CONNECTED = "bağlı"
DISCONNECTED = "bağlı değil"
HERMES_DOWN = "Hermes yanıt vermiyor"

MAX_SPOKEN_CHARS = 700  # per answer; the rest stays on screen ("Devamı ekranda.")


class _FrameWatcher(BaseObserver):
    def __init__(self, core: CelalCore, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self.core = core

    async def on_push_frame(self, data: FramePushed):
        f = data.frame
        c = self.core
        if isinstance(f, BotStartedSpeakingFrame):
            c._bot_speaking = True
            c._on_first_audio()
        elif isinstance(f, BotStoppedSpeakingFrame):
            c._bot_speaking = False
            c._last_audio = time.monotonic()
        elif isinstance(f, InterruptionFrame):
            if c._bot_speaking:
                c.interruptions += 1
        elif isinstance(f, UserStartedSpeakingFrame):
            c._user_speaking = True
        elif isinstance(f, UserStoppedSpeakingFrame):
            c._user_speaking = False
        else:
            return
        c._refresh()


class _Speech:
    """The answer currently being spoken (one tab, one turn)."""

    def __init__(self, tab_id: str):
        self.tab_id = tab_id
        self.buf = SentenceBuffer()
        self.chars = 0
        self.muted = False
        self.said_anything = False
        self.t0 = time.perf_counter()
        self.first_text: float | None = None
        self.first_audio: float | None = None
        self.total: float | None = None


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
        brain_factory=None,
        mac_api: bool | None = None,
        reverse: ReverseTunnel | None = None,
        warm_tabs: bool | None = None,
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
        self.listen = ListenState(active=not cfg.wake, on_change=self._on_wake)
        self.state = IDLE
        self.built: Built | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []
        self._runner_task: asyncio.Task | None = None
        self._bot_speaking = False
        self._user_speaking = False
        self._last_audio = 0.0
        self._last_speak = 0.0
        self._last_status: dict | None = None
        self.interruptions = 0
        self.last_error: str | None = None
        self.warm_tabs = settings.mac_warm if warm_tabs is None else warm_tabs
        if brain_factory is None:
            scrub_parent_env()  # the Mac Claude must not inherit a parent Claude session / API billing
            brain_factory = make_brain_factory(settings, self.client, self._ensure_connected)
        self.tabs = TabManager(
            brain_factory,
            max_parallel=settings.mac_max_parallel,
            state_dir=settings.state_dir,
            default_cwd=settings.mac_cwd,
            idle_close=settings.mac_idle_close,
            turn_timeout=settings.mac_timeout,
            on_event=self._on_tab,
        )
        self._speech: _Speech | None = None
        self._voice_loading: str | None = None
        self._timing_pending: _Speech | None = None
        self.timings: list[dict] = []  # per answer: first_text / first_audio / total seconds
        self.mac_api_enabled = settings.mac_api if mac_api is None else mac_api
        self.mac_api: MacApiServer | None = None
        self.mac_api_status = "kapalı"
        self.reverse = reverse
        if self.reverse is None and self.mac_api_enabled:
            self.reverse = ReverseTunnel(settings)
        if self.reverse is not None:
            self.reverse._on_status = self._on_tunnel

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
            "wake": {"on": self.cfg.wake, "active": self.listen.active, "start": START_PHRASE, "stop": STOP_PHRASE},
            "tts": self._tts_view(),
            "conversation": self.client.conversation,
            "tabs": self.tabs.views(),
            "active": self.tabs.active_id,
            "mac": {
                "running": len(self.tabs.running),
                "queued": len(self.tabs.queued),
                "max": self.tabs.max_parallel,
                "api": self.mac_api_status,
                "reverse": (f"{self.reverse.status}" if self.reverse is not None else "kapalı"),
            },
        }

    def _tts_view(self) -> dict | None:
        tts = self.built.tts if self.built else None
        if tts is None or not hasattr(tts, "voices"):
            return None
        return {"id": tts.voice_id, "voices": tts.voices(), "loading": self._voice_loading}

    def history(self) -> list[dict]:
        """Every tab's stored lines, for a page that (re)loads."""
        return [{"tab_id": t.id, "lines": t.lines[-50:], "partial": "".join(t.reply) if t.status == RUNNING else ""}
                for t in self.tabs.tabs.values()]

    def _compute_state(self) -> str:
        window = self.built.window if self.built else None
        if self._bot_speaking or (window is not None and window.closed):
            return SPEAKING
        active = self.tabs.active
        if active is not None and active.status in (QUEUED, RUNNING):
            return THINKING
        if self.mic:
            return LISTENING if self.listen.active else WAITING
        return IDLE

    def _refresh(self) -> None:
        self.state = self._compute_state()
        snap = self.snapshot()
        if snap != self._last_status:
            self._last_status = snap
            self.emit(snap)

    def _on_tunnel(self, status: str, detail: str) -> None:
        self._refresh()

    def _on_wake(self, active: bool) -> None:
        text = (
            f"Dinleme açık. “{STOP_PHRASE}” deyince kapanır."
            if active
            else f"Dinleme kapalı. “{START_PHRASE}” deyince açılır."
        )
        self.emit({"type": "wake", "active": active, "text": text})
        self._refresh()

    # ---- tabs ---------------------------------------------------------
    def _on_tab(self, kind: str, tab: Tab, data) -> None:
        if kind == "tabs":
            self._refresh()
        elif kind == "line":
            self.emit({"type": "tab_line", "tab_id": tab.id, "kind": tab.kind, "line": data})
        elif kind == "delta":
            self.emit({"type": "tab_delta", "tab_id": tab.id, "text": data})
            self._speech_delta(tab, data)
        elif kind == "break":
            self.emit({"type": "tab_break", "tab_id": tab.id})
            self._speech_delta(tab, "\n")
        elif kind == "turn_start":
            self.emit({"type": "tab_turn", "tab_id": tab.id, "phase": "start"})
            if tab.id == self.tabs.active_id:
                self._speech = _Speech(tab.id)
        elif kind == "turn_end":
            self._speech_end(tab, data)
            self.emit({"type": "tab_turn", "tab_id": tab.id, "phase": "end", "status": tab.status})

    def _speech_delta(self, tab: Tab, text: str) -> None:
        sp = self._speech
        if sp is None or sp.tab_id != tab.id:
            return
        if sp.first_text is None and text.strip():
            sp.first_text = time.perf_counter()
        if sp.muted:
            return
        for sentence in sp.buf.feed(text):
            self._say_sentence(sp, sentence)

    def _say_sentence(self, sp: _Speech, sentence: str) -> None:
        if sp.chars >= MAX_SPOKEN_CHARS:
            sp.muted = True
            self._queue_tts("Devamı ekranda.")
            return
        sp.chars += len(sentence)
        sp.said_anything = True
        self._queue_tts(sentence)

    def _speech_end(self, tab: Tab, outcome: TurnOutcome) -> None:
        sp = self._speech
        if sp is not None and sp.tab_id == tab.id:
            self._speech = None
            if not sp.muted and not outcome.interrupted:
                for sentence in sp.buf.flush():
                    self._say_sentence(sp, sentence)
                if not sp.said_anything:
                    if not outcome.ok and outcome.error:
                        self._queue_tts(f"Hata: {spoken_summary(outcome.error, 1, 160)}")
                    elif outcome.text:
                        self._queue_tts(spoken_summary(outcome.text))
            sp.total = time.perf_counter() - sp.t0
            if sp.said_anything and sp.first_audio is None and not outcome.interrupted:
                self._timing_pending = sp  # reported when its first audio plays
            else:
                self._report_timing(sp)
            return
        if tab.id != self.tabs.active_id and tab.source == "celal" and not outcome.interrupted:
            # A background tab finished: one short sentence, the tab gets an unread dot.
            verb = "bitti" if outcome.ok else "hata verdi"
            self._queue_tts(f"{tab.label} görevi {verb}.")

    def _on_first_audio(self) -> None:
        sp = self._speech
        if sp is not None and sp.first_audio is None and sp.said_anything:
            sp.first_audio = time.perf_counter()
        pending, self._timing_pending = self._timing_pending, None
        if pending is not None:
            pending.first_audio = time.perf_counter()
            self._report_timing(pending)

    def _report_timing(self, sp: _Speech) -> None:
        parts = []
        if sp.first_text is not None:
            parts.append(f"ilk kelime {sp.first_text - sp.t0:.1f} s")
        if sp.first_audio is not None:
            parts.append(f"ilk ses {sp.first_audio - sp.t0:.1f} s")
        if sp.total is not None:
            parts.append(f"tam cevap {sp.total:.1f} s")
        self.timings.append({"first_text": sp.first_text and sp.first_text - sp.t0,
                             "first_audio": sp.first_audio and sp.first_audio - sp.t0, "total": sp.total})
        self.emit({"type": "timing", "tab_id": sp.tab_id, "text": "⏱ " + ", ".join(parts)})

    def _queue_tts(self, text: str) -> None:
        if self.built is None or not text.strip():
            return
        self._last_speak = time.monotonic()
        self._tasks.append(asyncio.ensure_future(self.built.worker.queue_frame(TTSSpeakFrame(text=text))))

    @property
    def speaking(self) -> bool:
        """Audio is playing, or a sentence was queued less than 1.5 s ago and has not started yet."""
        return self._bot_speaking or (time.monotonic() - max(self._last_speak, 0) < 1.5
                                      and self._last_speak > self._last_audio)

    async def wait_quiet(self, timeout: float = 120.0) -> bool:
        """Until nothing is spoken any more (text mode)."""
        deadline = time.monotonic() + timeout
        quiet_since = None
        while time.monotonic() < deadline:
            if self.speaking:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since > 1.2:
                return True
            await asyncio.sleep(0.1)
        return False

    async def _cut_speech(self) -> bool:
        """Silence now; the current answer stays unspoken. True if something was (about to be) spoken."""
        was = self.speaking
        if self._speech is not None:
            self._speech.muted = True
        if self.built is not None:
            await self.built.worker.queue_frame(InterruptionWorkerFrame(), FrameDirection.UPSTREAM)
        self._last_speak = 0.0
        return was

    # ---- routing ------------------------------------------------------
    def _ensure_connected(self):
        async def check() -> bool:
            if self.connected:
                return True
            ok, reason = await self.client.ping()
            self.connected, self.reason = ok, reason
            self._refresh()
            return ok

        return check()

    def _activate(self, tab: Tab) -> None:
        if self.tabs.active_id != tab.id:
            if self._speech is not None and self._speech.tab_id != tab.id:
                self._tasks.append(asyncio.ensure_future(self._cut_speech()))
                self._speech = None
            self.tabs.set_active(tab.id)

    def _new_tab(self, kind: str, text: str, *, cwd: str | None = None) -> Tab:
        label = make_label(text, 3) if kind != MAC else make_label(text)
        if kind != MAC:
            label = f"Hermes: {label}"
        tab = self.tabs.create(kind, label=label, cwd=cwd, activate=False)
        self._activate(tab)
        return tab

    async def send_text(self, text: str, *, source: str = "text") -> Tab | None:
        """Route a typed or spoken message to a tab (see the module docstring); None if nothing was sent."""
        text = text.strip()
        if not text:
            return None
        target, body = route(text, hermes_prefixes=self.settings.hermes_prefixes,
                             new_tab_prefixes=self.settings.new_tab_prefixes,
                             mac_prefixes=self.settings.mac_prefixes)
        active = self.tabs.active
        new = target == NEW
        if new:
            target, body = route(body, hermes_prefixes=self.settings.hermes_prefixes,
                                 mac_prefixes=self.settings.mac_prefixes)
        kind = "hermes" if target == HERMES else MAC
        if new:
            tab = None
        elif target == HERMES:
            tab = active if active is not None and active.kind == "hermes" else None
        elif active is not None and active.kind == MAC:
            tab = active  # the default: the Mac's Claude Code in the active tab
        else:
            # Mac first: an unprefixed message never goes to Hermes, even from a Hermes tab.
            # Back to the latest idle Mac tab, else a new one.
            idle = [t for t in self.tabs.tabs.values() if t.kind == MAC and not t.busy]
            tab = max(idle, key=lambda t: t.updated) if idle else None
        _, cwd = split_cwd(body) if kind == MAC else (body, None)
        if tab is not None and cwd and tab.turns == 0 and not tab.lines and not same_dir(cwd, tab.cwd):
            # A fresh tab, but the first message names another folder: start Claude there instead.
            empty, tab = tab, None
            await self.tabs.close(empty.id)
        if tab is None:
            tab = self._new_tab(kind, body, cwd=cwd)
        self._activate(tab)
        logger.info(f"[{tab.kind}] sekme “{tab.label}” ← {body[:80]}")
        self.tabs.submit(tab, body)
        self._refresh()
        return tab

    def _route_spoken(self, text: str) -> None:
        """Voice pipeline (TurnSink): a finished spoken turn."""
        self._tasks.append(asyncio.ensure_future(self.send_text(text, source="mic")))

    async def new_tab(self) -> Tab:
        tab = self.tabs.create(MAC, activate=False)
        self._activate(tab)
        if self.warm_tabs:
            self.tabs.warm(tab)
        self._refresh()
        return tab

    async def select_tab(self, tab_id: str) -> None:
        tab = self.tabs.get(tab_id)
        if tab is None:
            return
        self._activate(tab)
        if self.warm_tabs and tab.kind == MAC and tab.brain is None and not tab.busy:
            self.tabs.warm(tab)
        self._refresh()

    async def close_tab(self, tab_id: str) -> None:
        if self._speech is not None and self._speech.tab_id == tab_id:
            await self._cut_speech()
            self._speech = None
        await self.tabs.close(tab_id)
        if not self.tabs.tabs:
            await self.new_tab()
        self._refresh()

    # ---- Durdur -------------------------------------------------------
    async def interrupt(self) -> dict:
        """Durdur: silence now, stop the active tab's Mac task. Hermes / server work is never cancelled."""
        tab = self.tabs.active
        tab_id = tab.id if tab else None
        self.emit({"type": "stopping", "tab_id": tab_id})
        spoke = await self._cut_speech()
        result: dict = {"speech": spoke, "stopped": False, "dropped": 0, "server": False}
        if tab is not None and tab.busy:
            if tab.kind == MAC:
                result.update(await self.tabs.stop(tab))
            else:
                result.update(await self.tabs.stop(tab))  # drops its queued messages only
                result["server"] = True
        parts = []
        if result.get("stopped"):
            parts.append("Mac görevi durduruldu")
        if result.get("server"):
            parts.append("sunucudaki iş sürüyor (Durdur sunucuya iptal göndermez)")
        if spoke:
            parts.append("ses kesildi")
        if result.get("dropped"):
            parts.append(f"sıradaki {result['dropped']} mesaj iptal")
        text = ("Durdur: " + ", ".join(parts) + ".") if parts else "Durduracak bir şey yoktu."
        text = text[0].upper() + text[1:]
        result["text"] = text
        self.emit({"type": "stopped", "tab_id": tab_id, "text": text})
        self._refresh()
        return result

    # ---- lifecycle ----------------------------------------------------
    async def start(self, warm: bool = True) -> None:
        self.loop = asyncio.get_running_loop()
        self.tunnel.start()
        restored = self.tabs.restore()
        if not self.tabs.tabs:
            self.tabs.create(MAC)
        logger.info(f"sekmeler: {len(self.tabs.tabs)} ({restored} geri geldi)")
        cfg = self.cfg
        if not self.voice:
            self.built = build_text(cfg, observers=[_FrameWatcher(self)])
        else:
            listen = self.listen if cfg.wake else None
            self.built = build_voice(cfg, self._route_spoken, observers=[_FrameWatcher(self)], listen=listen)
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
        active = self.tabs.active
        if self.warm_tabs and active is not None and active.kind == MAC:
            self.tabs.warm(active)  # Claude's CLI starts now, not at the first message
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
            self.mac_api = MacApiServer(self.tabs, token, port=self.settings.mac_api_port,
                                        on_request=self._on_api_request)
            await self.mac_api.start()
            self.mac_api_status = f"127.0.0.1:{self.mac_api.port}"
        except OSError as e:
            self.mac_api = None
            self.mac_api_status = "açılamadı"
            self.emit({"type": "error", "text": f"Mac API açılamadı (127.0.0.1:{self.settings.mac_api_port}): "
                       f"{e.strerror or e}. Hermes → Mac kapalı; başka bir Celal açık olabilir."})
            return
        if self.reverse is not None:
            self.reverse.start()

    def _on_api_request(self, kind: str, data: dict) -> None:
        if kind == "task":
            logger.info(f"[mac] Hermes'ten görev (📱 {data.get('tab_id')}): {str(data.get('task'))[:80]}")
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

    def kick_threshold(self) -> int:
        """Failed pings before the next tunnel restart: 3, 6, 12, … (capped)."""
        return self.stall_after * (2 ** min(self.kicks_since_ok, self.max_kick_backoff))

    async def _health_loop(self) -> None:
        fails = 0
        while True:
            ok, reason = await self.client.ping()
            if not ok and self.tunnel.status in (OPEN, EXTERNAL):
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
                    self.tunnel.kick("(sağlık kontrolü cevapsız)")
                    self.kicks_since_ok += 1
                    fails = 0
            await asyncio.sleep(self.health_interval)

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(0.3)
            self._refresh()

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
                await self.tabs.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"tabs shutdown: {e}")
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
    async def set_voice(self, voice_id: str) -> None:
        """Switch the speaking voice (window menu); loading a big model takes a while."""
        tts = self.built.tts if self.built else None
        if tts is None or voice_id == tts.voice_id or self._voice_loading:
            self._refresh()
            return
        voice = BY_ID.get(voice_id)
        label = voice.label if voice else voice_id
        self._voice_loading = voice_id
        self._refresh()
        try:
            await tts.select(voice_id)
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "error", "text": f"{label} açılamadı: {e}"})
        else:
            save_choice(self.settings.state_dir, voice_id)
            self.emit({"type": "info", "text": f"Ses: {label}"})
            self._queue_tts("Merhaba, artık bu sesle konuşuyorum.")
        finally:
            self._voice_loading = None
            self._refresh()

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
        if self.cfg.wake:
            self.listen.set_active(False)  # every mic open starts passive, like a fresh "Hey Siri"
        self._refresh()

    # ---- thread bridge (UI threads) -----------------------------------
    def call(self, coro) -> None:
        """Schedule a coroutine on the core loop from another thread."""
        if self.loop is None:
            coro.close()
            return
        asyncio.run_coroutine_threadsafe(coro, self.loop)
