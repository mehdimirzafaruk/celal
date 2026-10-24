"""Tabs: one task / conversation each, up to ``max_parallel`` answering at once.

Every tab owns a *brain* (:mod:`celal.brains`: the Mac's Claude Code by
default, Hermes for "sunucuda …" tabs) and one worker task. Messages for a
tab go into its own queue and are answered in order; a global semaphore
lets at most ``max_parallel`` tabs answer at the same time (the rest show
"sırada"). The worker is the only task that connects, queries and closes its
brain (the Claude SDK client must stay in the task that connected it); a tab
whose Claude client sat idle for ``idle_close`` seconds closes it and keeps
the session id, so the next message resumes the same conversation.

Durdur (:meth:`TabManager.stop`) interrupts a Mac tab's current answer and
drops its queued messages. Hermes tabs are never interrupted from here:
work sent to the server keeps running (the caller only cuts the speech).

Tabs survive restarts in ``tabs.json`` (label, session id, cwd, last status,
last 50 lines). Tasks started through the Mac API (:mod:`celal.mac_api`) are
also kept in ``tasks.json`` for 24 h by :class:`TaskStore`.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from celal.brains import Brain, TurnOutcome

IDLE = "idle"
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
STOPPED = "stopped"

MAC = "mac"
HERMES = "hermes"

KEEP_LINES = 50
TASK_TTL = 24 * 3600
_WARM = object()


def make_label(text: str, words: int = 4, max_chars: int = 28) -> str:
    """First few words of a task: "README dosyasını güncelle ve" → tab label."""
    flat = " ".join(str(text).split())
    label = " ".join(flat.split(" ")[:words])
    if len(label) > max_chars:
        label = label[: max_chars - 1].rstrip() + "…"
    return label or "Yeni görev"


@dataclass
class Tab:
    id: str
    label: str
    kind: str = MAC  # "mac" | "hermes"
    source: str = "celal"  # "celal" (window / terminal) | "hermes" (Mac API: 📱)
    cwd: str = "~"
    session_id: str | None = None
    status: str = IDLE
    lines: list[dict] = field(default_factory=list)
    unread: bool = False
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    # runtime only
    queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)
    worker: asyncio.Task | None = field(default=None, repr=False)
    brain: Brain | None = field(default=None, repr=False)
    reply: list[str] = field(default_factory=list, repr=False)
    turn_started: float = 0.0
    turn_text: str = ""
    task_id: str | None = None  # Mac API task of the current turn
    stop_requested: bool = False
    stopping: bool = False
    closed: bool = False
    turn_done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    turns: int = 0
    finished: int = 0  # answers completed (any status)

    @property
    def waiting(self) -> int:
        """Messages queued behind the current one."""
        return self.queue.qsize()

    @property
    def busy(self) -> bool:
        return self.status in (QUEUED, RUNNING)

    def view(self, active: bool = False) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "source": self.source,
            "status": self.status,
            "stopping": self.stopping,
            "unread": self.unread,
            "active": active,
            "waiting": self.waiting,
            "cwd": self.cwd,
        }

    def to_json(self) -> dict:
        return {
            "id": self.id, "label": self.label, "kind": self.kind, "source": self.source, "cwd": self.cwd,
            "session_id": self.session_id, "status": self.status, "lines": self.lines[-KEEP_LINES:],
            "unread": self.unread, "created": self.created, "updated": self.updated,
        }

    @classmethod
    def from_json(cls, d: dict) -> Tab:
        t = cls(id=str(d["id"]), label=str(d.get("label") or "Görev"), kind=d.get("kind") or MAC,
                source=d.get("source") or "celal", cwd=d.get("cwd") or "~", session_id=d.get("session_id"),
                status=d.get("status") or IDLE, lines=list(d.get("lines") or [])[-KEEP_LINES:],
                unread=bool(d.get("unread")), created=float(d.get("created") or time.time()),
                updated=float(d.get("updated") or time.time()))
        if t.status in (QUEUED, RUNNING):  # Celal stopped while it worked
            t.status = STOPPED
            t.lines.append({"role": "info", "text": "Celal kapandığı için görev yarıda kaldı.", "t": time.time()})
        return t


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.warning(f"{path} okunamadı: {e}")
        return None


class TaskStore:
    """Mac API task records (``tasks.json``), kept 24 h for ``GET /task/<id>``."""

    def __init__(self, path: str | Path | None, ttl: float = TASK_TTL, clock: Callable[[], float] = time.time):
        self.path = Path(path).expanduser() if path else None
        self.ttl = ttl
        self._clock = clock
        self.records: dict[str, dict] = {}
        if self.path is not None:
            data = _read_json(self.path) or {}
            for rec in data.get("tasks", []) if isinstance(data, dict) else []:
                if isinstance(rec, dict) and rec.get("task_id"):
                    if rec.get("status") in (QUEUED, RUNNING):
                        rec["status"], rec["error"] = "error", "Celal yeniden başladı; görev yarıda kaldı"
                        rec["finished_at"] = rec.get("finished_at") or self._clock()
                    self.records[rec["task_id"]] = rec
            self._prune()

    def _prune(self) -> None:
        cutoff = self._clock() - self.ttl
        for k in [k for k, r in self.records.items() if (r.get("finished_at") or r.get("created_at") or 0) < cutoff
                  and r.get("status") not in (QUEUED, RUNNING)]:
            del self.records[k]

    def save(self) -> None:
        self._prune()
        if self.path is not None:
            try:
                _write_json(self.path, {"tasks": list(self.records.values())})
            except OSError as e:
                logger.warning(f"tasks.json yazılamadı: {e}")

    def create(self, tab: Tab, task: str, *, queued: bool) -> dict:
        rec = {
            "task_id": uuid.uuid4().hex[:10], "tab_id": tab.id, "task": task, "label": tab.label, "cwd": tab.cwd,
            "source": tab.source, "status": QUEUED if queued else RUNNING, "result": "", "error": "",
            "created_at": self._clock(), "started_at": None, "finished_at": None,
        }
        self.records[rec["task_id"]] = rec
        self.save()
        return rec

    def get(self, task_id: str) -> dict | None:
        rec = self.records.get(task_id)
        if rec is not None and rec.get("status") not in (QUEUED, RUNNING) and \
                (rec.get("finished_at") or 0) < self._clock() - self.ttl:
            return None
        return rec

    def update(self, task_id: str | None, **fields) -> None:
        rec = self.records.get(task_id or "")
        if rec is None:
            return
        rec.update(fields)
        self.save()

    def active(self) -> list[dict]:
        return [r for r in self.records.values() if r.get("status") in (QUEUED, RUNNING)]


class TabManager:
    """Owns the tabs, their workers and the shared ``max_parallel`` slots.

    ``on_event(kind, tab, data)`` is called on the event loop with kind:
    ``tabs`` (list or a status changed), ``line`` (a line was added: data is the
    line dict), ``delta`` (reply text), ``break`` (new text block),
    ``turn_start`` (data: the message), ``turn_end`` (data: :class:`TurnOutcome`).
    """

    def __init__(
        self,
        brain_factory: Callable[[Tab], Brain],
        *,
        max_parallel: int = 3,
        state_dir: str | Path | None = None,
        default_cwd: str = "~",
        idle_close: float = 600.0,
        turn_timeout: float = 1800.0,
        stop_grace: float = 8.0,
        on_event: Callable[[str, Tab, object], None] | None = None,
        tasks: TaskStore | None = None,
    ):
        self._factory = brain_factory
        self.max_parallel = max(1, int(max_parallel))
        self._slots = asyncio.Semaphore(self.max_parallel)
        self.state_path = Path(state_dir).expanduser() / "tabs.json" if state_dir else None
        self.default_cwd = default_cwd
        self.idle_close = idle_close
        self.turn_timeout = turn_timeout
        self.stop_grace = stop_grace
        self.on_event = on_event
        self.tasks = tasks or TaskStore(Path(state_dir).expanduser() / "tasks.json" if state_dir else None)
        self.tabs: dict[str, Tab] = {}
        self.active_id: str | None = None
        self._save_handle: asyncio.TimerHandle | None = None

    # ---- queries ------------------------------------------------------
    @property
    def active(self) -> Tab | None:
        return self.tabs.get(self.active_id or "")

    @property
    def running(self) -> list[Tab]:
        return [t for t in self.tabs.values() if t.status == RUNNING]

    @property
    def queued(self) -> list[Tab]:
        return [t for t in self.tabs.values() if t.status == QUEUED]

    def slot_free(self) -> bool:
        return len(self.running) + len(self.queued) < self.max_parallel

    def get(self, tab_id: str) -> Tab | None:
        return self.tabs.get(tab_id)

    def views(self) -> list[dict]:
        return [t.view(t.id == self.active_id) for t in self.tabs.values()]

    # ---- events / persistence ----------------------------------------
    def _emit(self, kind: str, tab: Tab, data=None) -> None:
        if tab.closed and kind != "tabs":
            return
        if self.on_event is not None:
            try:
                self.on_event(kind, tab, data)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"tab event failed: {e}")

    def _changed(self, tab: Tab) -> None:
        tab.updated = time.time()
        self._emit("tabs", tab)
        self._save_soon()

    def add_line(self, tab: Tab, role: str, text: str, **extra) -> dict:
        line = {"role": role, "text": text, "t": time.time(), **extra}
        tab.lines.append(line)
        if len(tab.lines) > KEEP_LINES * 2:
            del tab.lines[: len(tab.lines) - KEEP_LINES]
        self._emit("line", tab, line)
        self._save_soon()
        return line

    def _save_soon(self) -> None:
        if self.state_path is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.save()
            return
        if self._save_handle is None:
            self._save_handle = loop.call_later(0.5, self.save)

    def save(self) -> None:
        self._save_handle = None
        if self.state_path is None:
            return
        data = {"active": self.active_id, "tabs": [t.to_json() for t in self.tabs.values() if not t.closed]}
        try:
            _write_json(self.state_path, data)
        except OSError as e:
            logger.warning(f"tabs.json yazılamadı: {e}")

    def restore(self) -> int:
        if self.state_path is None:
            return 0
        data = _read_json(self.state_path)
        if not isinstance(data, dict):
            return 0
        for d in data.get("tabs") or []:
            try:
                t = Tab.from_json(d)
            except (KeyError, TypeError, ValueError):
                continue
            self.tabs[t.id] = t
        active = data.get("active")
        self.active_id = active if active in self.tabs else (next(iter(self.tabs), None))
        return len(self.tabs)

    # ---- tabs ---------------------------------------------------------
    def create(self, kind: str = MAC, *, source: str = "celal", label: str | None = None, cwd: str | None = None,
               activate: bool = True) -> Tab:
        tab = Tab(id=uuid.uuid4().hex[:8], label=label or "Yeni görev", kind=kind, source=source,
                  cwd=os.path.expanduser(cwd or self.default_cwd))
        self.tabs[tab.id] = tab
        if activate or self.active_id is None:
            self.active_id = tab.id
        self._changed(tab)
        return tab

    def set_active(self, tab_id: str) -> Tab | None:
        tab = self.tabs.get(tab_id)
        if tab is None:
            return None
        self.active_id = tab_id
        tab.unread = False
        self._changed(tab)
        return tab

    async def close(self, tab_id: str) -> bool:
        tab = self.tabs.get(tab_id)
        if tab is None:
            return False
        if tab.busy and tab.kind == MAC:
            await self.stop(tab)
        tab.closed = True
        ids = list(self.tabs)
        idx = ids.index(tab_id)
        del self.tabs[tab_id]
        self._drain(tab)
        if tab.kind == MAC or not tab.busy:
            # A Hermes tab still waiting on the server is left to finish on its own (no cancel to Hermes).
            if tab.worker is not None and not tab.worker.done():
                tab.worker.cancel()
        if self.active_id == tab_id:
            rest = [i for i in ids if i != tab_id]
            self.active_id = rest[min(idx, len(rest) - 1)] if rest else None
        self._changed(tab)
        return True

    # ---- messages -----------------------------------------------------
    def _ensure_worker(self, tab: Tab) -> None:
        if tab.worker is None or tab.worker.done():
            tab.worker = asyncio.create_task(self._worker(tab), name=f"tab-{tab.id}")

    def warm(self, tab: Tab) -> None:
        """Start the tab's brain ahead of the first message (Claude CLI start-up)."""
        if tab.kind != MAC or tab.closed:
            return
        tab.queue.put_nowait(_WARM)
        self._ensure_worker(tab)

    def submit(self, tab: Tab, text: str, *, task_id: str | None = None) -> str:
        """Queue a message for the tab; returns ``"running"`` or ``"queued"`` (no free slot / tab busy)."""
        text = text.strip()
        if not text:
            raise ValueError("mesaj boş")
        if tab.label == "Yeni görev" and tab.turns == 0:
            tab.label = make_label(text)
        will_wait = tab.busy or not self.slot_free()
        self.add_line(tab, "user", text, queued=will_wait)
        tab.queue.put_nowait((text, task_id))
        self._ensure_worker(tab)
        if not tab.busy:
            tab.status = QUEUED  # the worker flips it to running once it holds a slot
        self._changed(tab)
        return QUEUED if will_wait else RUNNING

    def _drain(self, tab: Tab) -> int:
        dropped = 0
        while not tab.queue.empty():
            item = tab.queue.get_nowait()
            if item is _WARM:
                continue
            dropped += 1
            self.tasks.update(item[1], status=STOPPED_API, error="durduruldu", finished_at=time.time())
        return dropped

    async def stop(self, tab: Tab | None = None) -> dict:
        """Durdur for one tab (default: the active one). Never touches Hermes tabs' server work."""
        tab = tab or self.active
        if tab is None:
            return {"stopped": False, "dropped": 0, "reason": "sekme yok"}
        if tab.kind != MAC:
            dropped = self._drain(tab)
            return {"stopped": False, "dropped": dropped, "server": tab.busy}
        dropped = self._drain(tab)
        if tab.status == QUEUED:
            # Still waiting for a slot: nothing runs yet, cancel the wait.
            tab.stop_requested = True
            if tab.worker is not None and not tab.worker.done():
                tab.worker.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(tab.worker), 2)
                except (asyncio.CancelledError, TimeoutError, Exception):  # noqa: BLE001
                    pass
            if tab.status == QUEUED:
                self._finish(tab, TurnOutcome(False, interrupted=True))
            return {"stopped": True, "dropped": dropped}
        if tab.status != RUNNING:
            return {"stopped": False, "dropped": dropped}
        tab.stop_requested = True
        tab.stopping = True
        self._changed(tab)
        brain = tab.brain
        if brain is not None:
            try:
                await asyncio.wait_for(brain.interrupt(), 5)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"interrupt: {e}")
        try:
            await asyncio.wait_for(tab.turn_done.wait(), self.stop_grace)
        except TimeoutError:
            # Claude did not end the turn: cancel the worker (it closes its client in its own task).
            if tab.worker is not None and not tab.worker.done():
                tab.worker.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(tab.worker), 5)
                except (asyncio.CancelledError, TimeoutError, Exception):  # noqa: BLE001
                    pass
            if tab.status == RUNNING:
                self._finish(tab, TurnOutcome(False, interrupted=True))
        return {"stopped": True, "dropped": dropped}

    # ---- worker -------------------------------------------------------
    def _brain(self, tab: Tab) -> Brain:
        if tab.brain is None:
            tab.brain = self._factory(tab)
        return tab.brain

    async def _worker(self, tab: Tab) -> None:
        try:
            while not tab.closed:
                brain = tab.brain
                connected = brain is not None and brain.connected
                getter = asyncio.ensure_future(tab.queue.get())
                waiters = {getter}
                unprompted = None
                if connected and hasattr(brain, "wait_message"):
                    # Claude may speak on its own between messages (a background task it left running finished).
                    unprompted = asyncio.ensure_future(brain.wait_message())
                    waiters.add(unprompted)
                try:
                    done, _ = await asyncio.wait(
                        waiters, timeout=self.idle_close if connected else None, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    for w in waiters:
                        if not w.done():
                            w.cancel()
                if getter in done:
                    item = getter.result()
                elif unprompted is not None and unprompted in done:
                    await self._turn(tab, None, None)
                    continue
                else:
                    logger.info(f"sekme {tab.label}: {int(self.idle_close)} s boşta, Claude kapatılıyor (oturum saklı)")
                    await brain.close()
                    continue
                if item is _WARM:
                    try:
                        await self._brain(tab).warm()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"sekme {tab.label}: Claude ısınamadı: {e}")
                    continue
                text, task_id = item
                await self._turn(tab, text, task_id)
        finally:
            brain, tab.brain = tab.brain, None
            if brain is not None:
                if brain.session_id:
                    tab.session_id = brain.session_id
                try:
                    await brain.close()  # in this task: the SDK client must close where it connected
                except BaseException as e:  # noqa: BLE001
                    logger.debug(f"brain close: {e}")

    async def _turn(self, tab: Tab, text: str | None, task_id: str | None) -> None:
        tab.task_id = task_id
        tab.turn_text = text or ""
        tab.stop_requested = False
        tab.turn_done.clear()
        if tab.status != QUEUED:
            tab.status = QUEUED
            self._changed(tab)
        try:
            async with self._slots:
                tab.status = RUNNING
                tab.turn_started = time.time()
                tab.reply = []
                tab.turns += 1
                self.tasks.update(task_id, status=RUNNING, started_at=tab.turn_started)
                self._changed(tab)
                self._emit("turn_start", tab, text)
                brain = self._brain(tab)

                def emit(kind: str, value: str) -> None:
                    self._on_brain(tab, kind, value)

                try:
                    async with asyncio.timeout(self.turn_timeout):
                        outcome = await brain.turn(text, emit)
                except TimeoutError:
                    await brain.close()
                    outcome = TurnOutcome(False, error=f"zaman aşımı: {int(self.turn_timeout // 60)} dk doldu")
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.exception("sekme cevabı başarısız")
                    await brain.close()
                    outcome = TurnOutcome(False, error=f"{type(e).__name__}: {e}"[:300])
                if brain.session_id:
                    tab.session_id = brain.session_id
                self._finish(tab, outcome)
        except asyncio.CancelledError:
            if tab.status in (QUEUED, RUNNING):
                self._finish(tab, TurnOutcome(False, interrupted=True))
            raise

    def _on_brain(self, tab: Tab, kind: str, value: str) -> None:
        if kind == "text":
            tab.reply.append(value)
            self._emit("delta", tab, value)
        elif kind == "break":
            if tab.reply and not tab.reply[-1].endswith("\n"):
                tab.reply.append("\n\n")
            self._emit("break", tab, "")
        elif kind == "progress":
            self.add_line(tab, "progress", value)
        elif kind == "session":
            tab.session_id = value
            self._save_soon()

    def _finish(self, tab: Tab, outcome: TurnOutcome) -> None:
        stopped = tab.stop_requested or outcome.interrupted
        status = STOPPED if stopped else (DONE if outcome.ok else ERROR)
        streamed = "".join(tab.reply).strip()
        text = streamed or (outcome.text or "").strip()
        seconds = round(time.time() - tab.turn_started, 1) if tab.turn_started else 0.0
        if text:
            self.add_line(tab, "assistant", text, live=bool(streamed), seconds=seconds)
        if status == ERROR:
            self.add_line(tab, "error", outcome.error or "hata")
        elif status == STOPPED:
            self.add_line(tab, "info", "durduruldu")
        api_status = {DONE: "done", ERROR: "error", STOPPED: STOPPED_API}[status]
        self.tasks.update(tab.task_id, status=api_status, result=(outcome.text or text) if status == DONE else text,
                          error=outcome.error if status == ERROR else ("durduruldu" if status == STOPPED else ""),
                          finished_at=time.time())
        tab.status = status
        tab.stopping = False
        tab.stop_requested = False
        tab.task_id = None
        tab.finished += 1
        if tab.id != self.active_id:
            tab.unread = True
        tab.turn_done.set()
        self._changed(tab)
        self._emit("turn_end", tab, outcome if not stopped else TurnOutcome(False, text, interrupted=True))

    async def shutdown(self) -> None:
        for tab in list(self.tabs.values()):
            if tab.worker is not None and not tab.worker.done():
                tab.worker.cancel()
        workers = [t.worker for t in self.tabs.values() if t.worker is not None]
        if workers:
            await asyncio.wait(workers, timeout=10)
        if self._save_handle is not None:
            self._save_handle.cancel()
        self.save()
        self.tasks.save()


STOPPED_API = "cancelled"  # the Mac API calls a stopped task "cancelled"
