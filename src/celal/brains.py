"""What answers a tab: the Mac's own Claude Code (default) or Hermes on the server.

A *brain* lives inside one tab's worker task (:mod:`celal.tabs`) and answers
one message at a time through :meth:`turn`, reporting as it goes through
``emit(kind, value)``:

- ``("text", delta)``: reply text as it streams (shown, and spoken when the tab is active)
- ``("break", "")``: a new text block starts (a sentence boundary for speech)
- ``("progress", line)``: a short Turkish step line ("komut çalıştırıyor: pytest"), never spoken
- ``("session", id)``: the Claude session id (saved so the tab can resume later)

:class:`MacClaudeBrain` keeps one persistent ``ClaudeSDKClient`` per tab
(claude-agent-sdk): follow-up messages go to the same process, so they skip
the CLI start-up. It runs with ``permission_mode="bypassPermissions"`` and the
user's own ``~/.claude`` settings, skills, hooks, MCP servers and CLAUDE.md
(``setting_sources`` left at ``None``: every source is loaded). The SDK
requires connect, queries and disconnect to happen in the task that
connected, which the tab worker guarantees; :meth:`MacClaudeBrain.interrupt`
is the only call made from another task (Durdur).

:class:`HermesBrain` streams one request to Hermes. Durdur never cancels it
(server work is not stopped from the Mac); only the speech is cut.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from celal.mac_task import child_env, describe_tool, resolve_claude

Emit = Callable[[str, str], None]

MAC_PROMPT = (
    "Celal sesli asistanından konuşuyorsun; cevabın Mac'te sesli okunacak. Türkçe ve kısa cevap ver: "
    "çoğunlukla 1-3 cümle, en önemli bilgiyi (sonuç, sayı, hata) başa koy. Markdown, tablo ve emoji kullanma. "
    "Mac'te yapamadığın, gerçekten sunucuda olması gereken işler için (Hermes cron'ları, Yasomi içerik hattı, "
    "sunucudaki servisler, Telegram'a mesaj) Bash ile `{hermes_cmd} \"<istek>\"` komutunu çalıştır "
    "(uzun sürebilir: Bash timeout'unu 600000 ver); cevabı Hermes'ten gelir. Sunucuya SSH gerekiyorsa "
    "hermes-server skill'ini kullan. Mac'te yapılabilecek işi asla sunucuya gönderme."
)

# Variables of a Claude Code session Celal may have been started from. Passed on,
# they would make the child think it runs inside that session, bill the API
# (ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL) instead of the Max login, or reuse its ids.
_KEEP_ENV = {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"}


def scrub_parent_env(env: dict[str, str] | None = None) -> list[str]:
    """Drop ``CLAUDE*`` / ``ANTHROPIC_*`` from Celal's own environment; returns the removed names.

    The SDK builds the child's environment from ``os.environ`` (it only drops
    ``CLAUDECODE``), so the cleanup has to happen here, once.
    """
    env = os.environ if env is None else env
    removed = [k for k in list(env) if k not in _KEEP_ENV and (k.startswith("CLAUDE") or k.startswith("ANTHROPIC_"))]
    for k in removed:
        env.pop(k, None)
    return removed


def celal_command() -> str:
    """How the Mac Claude reaches Hermes: ``celal`` from this venv (``uv run`` puts it on PATH)."""
    exe = Path(sys.executable).parent / "celal"
    return str(exe) + " hermes" if exe.exists() else "celal hermes"


@dataclass
class TurnOutcome:
    ok: bool
    text: str = ""
    error: str = ""
    interrupted: bool = False
    session_id: str | None = None


class Brain:
    """Interface (duck-typed): see the module docstring."""

    kind = "mac"
    stoppable = True  # Durdur may interrupt it
    session_id: str | None = None

    @property
    def connected(self) -> bool:
        return False

    async def warm(self) -> None:  # start the process ahead of the first message
        pass

    async def turn(self, text: str, emit: Emit) -> TurnOutcome:  # pragma: no cover - interface
        raise NotImplementedError

    async def interrupt(self) -> None:
        pass

    async def close(self) -> None:
        pass


_EOF = object()
_ABORT = object()
TERMINAL = {"completed", "failed", "stopped", "killed"}
FOLLOWUP_WAIT = 60.0  # seconds to wait for Claude's report after a background task ended


class _Inbox:
    """FIFO of SDK messages that can be waited on without taking an item."""

    def __init__(self):
        self._items: deque = deque()
        self._event = asyncio.Event()

    def put(self, item) -> None:
        self._items.append(item)
        self._event.set()

    def empty(self) -> bool:
        return not self._items

    async def wait(self, pred=None) -> None:
        """Until an item is waiting (one matching ``pred`` when given)."""
        while not (any(pred(i) for i in self._items) if pred else self._items):
            self._event.clear()
            await self._event.wait()

    async def get(self):
        await self.wait()
        return self._items.popleft()

    def drop_aborts(self) -> None:
        self._items = deque(i for i in self._items if i is not _ABORT)


class MacClaudeBrain(Brain):
    """The Mac's Claude Code through one persistent ``ClaudeSDKClient``.

    A reader task moves every SDK message into an inbox. :meth:`turn` sends
    the message and consumes the inbox until Claude's answer is complete:
    the result of our query arrived *and* no background task it started
    (``run_in_background`` shells, agents) is still running; their
    notification turns are part of the answer. Messages that arrive between
    turns (a background task Claude left running finishes later) make
    :attr:`has_message` true; the tab worker then calls ``turn(None, …)`` to
    read that unprompted answer.
    """

    kind = "mac"
    stoppable = True

    def __init__(
        self,
        *,
        cwd: str,
        model: str | None = "sonnet",
        claude: str = "claude",
        session_id: str | None = None,
        append_prompt: str | None = None,
        client_factory: Callable[[object], object] | None = None,
        options_factory: Callable[..., object] | None = None,
        abort_grace: float = 3.0,
    ):
        self.cwd = os.path.expanduser(cwd)
        self.model = model or None
        self.claude = claude
        self.session_id = session_id
        self.append_prompt = append_prompt or MAC_PROMPT.format(hermes_cmd=celal_command())
        self._client_factory = client_factory
        self._options_factory = options_factory
        self.abort_grace = abort_grace
        self._client = None
        self._reader: asyncio.Task | None = None
        self._inbox = _Inbox()
        self._said_model = False
        self._aborting = False

    @property
    def connected(self) -> bool:
        return self._client is not None

    @property
    def has_message(self) -> bool:
        return not self._inbox.empty()

    async def wait_message(self) -> None:
        """Returns when Claude starts answering on its own (used by an idle tab worker).

        System messages (hook output, status) that arrive between turns do not
        count; they are read at the start of the next turn.
        """
        from claude_agent_sdk import SystemMessage

        await self._inbox.wait(lambda m: not isinstance(m, SystemMessage) and m is not _ABORT)

    def options(self):
        env = child_env()
        # The venv's bin first: `celal hermes` resolves even when Celal was not started through uv.
        venv_bin = str(Path(sys.executable).parent)
        if venv_bin not in env.get("PATH", "").split(":"):
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        kwargs = dict(
            cwd=self.cwd,
            model=self.model,
            permission_mode="bypassPermissions",
            include_partial_messages=True,
            system_prompt={"type": "preset", "preset": "claude_code", "append": self.append_prompt},
            resume=self.session_id,
            env={"PATH": env["PATH"]},
            stderr=lambda line: logger.debug(f"claude: {line.rstrip()}"),
        )
        cli = resolve_claude(self.claude, env)
        if cli:
            kwargs["cli_path"] = cli  # the user's own (updated) Claude Code; else the SDK's bundled one
        if self._options_factory is not None:
            return self._options_factory(**kwargs)
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(**kwargs)

    async def _connect(self) -> None:
        if self._client is not None:
            return
        opts = self.options()
        if self._client_factory is not None:
            client = self._client_factory(opts)
        else:
            from claude_agent_sdk import ClaudeSDKClient

            client = ClaudeSDKClient(opts)
        await client.connect()
        self._client = client
        self._inbox = _Inbox()
        self._reader = asyncio.create_task(self._read(client), name="claude-reader")

    async def _read(self, client) -> None:
        try:
            async for msg in client.receive_messages():
                self._inbox.put(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._inbox.put(e)
        finally:
            self._inbox.put(_EOF)

    async def warm(self) -> None:
        await self._connect()

    async def turn(self, text: str | None, emit: Emit) -> TurnOutcome:
        """Answer ``text``; with ``None`` read an unprompted answer already waiting in the inbox."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            TaskNotificationMessage,
            TaskStartedMessage,
            TaskUpdatedMessage,
            TextBlock,
            ToolUseBlock,
        )

        await self._connect()
        self._aborting = False
        self._inbox.drop_aborts()  # a Durdur that came after the previous turn had already ended
        if text is not None:
            await self._client.query(text)
        streamed = 0  # text deltas since the last full assistant message
        pending: set[str] = set()  # background tasks started in this answer
        outcome: TurnOutcome | None = None
        result_text = ""
        followup = False  # a background task ended after our result: Claude reports on it in a new turn
        while True:
            if outcome is not None and not pending and not followup:
                return outcome
            try:
                msg = await asyncio.wait_for(self._inbox.get(), FOLLOWUP_WAIT if followup and not pending else None)
            except TimeoutError:
                return outcome
            if msg is _ABORT:
                return await self._abort(outcome, pending)
            if msg is _EOF or isinstance(msg, Exception):
                self._client = None  # the CLI is gone; the next message reconnects (resume)
                if outcome is not None:
                    return outcome
                if self._aborting:
                    return TurnOutcome(False, interrupted=True, session_id=self.session_id)
                raise RuntimeError(f"Claude bağlantısı koptu{': ' + str(msg) if isinstance(msg, Exception) else ''}")
            if isinstance(msg, StreamEvent):
                if msg.parent_tool_use_id is not None:
                    continue  # a subagent's own stream
                ev = msg.event or {}
                kind = ev.get("type")
                if kind == "content_block_start" and (ev.get("content_block") or {}).get("type") == "text":
                    emit("break", "")
                elif kind == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        streamed += len(delta["text"])
                        emit("text", delta["text"])
            elif isinstance(msg, AssistantMessage):
                if msg.parent_tool_use_id is not None:
                    continue
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        emit("progress", describe_tool(block.name, block.input))
                    elif isinstance(block, TextBlock) and block.text and not streamed:
                        emit("break", "")
                        emit("text", block.text)  # no partial events came for it
                streamed = 0
            elif isinstance(msg, TaskStartedMessage):
                pending.add(msg.task_id)
                emit("progress", f"arka planda: {msg.description[:60]}")
            elif isinstance(msg, TaskNotificationMessage):
                if msg.task_id in pending and outcome is not None:
                    followup = True
                pending.discard(msg.task_id)
            elif isinstance(msg, TaskUpdatedMessage):
                status = msg.status or (msg.patch or {}).get("status")
                if status in TERMINAL:
                    if msg.task_id in pending and outcome is not None:
                        followup = True
                    pending.discard(msg.task_id)
            elif isinstance(msg, SystemMessage):
                if msg.subtype == "init":
                    sid = (msg.data or {}).get("session_id")
                    if sid and sid != self.session_id:
                        self.session_id = sid
                        emit("session", sid)
                    if not self._said_model:
                        self._said_model = True
                        emit("progress", f"Claude hazır ({(msg.data or {}).get('model', '?')})")
            elif isinstance(msg, ResultMessage):
                if msg.session_id and msg.session_id != self.session_id:
                    self.session_id = msg.session_id
                    emit("session", msg.session_id)
                aborted = (msg.terminal_reason or "").startswith("aborted")
                if not pending:
                    followup = False
                result = msg.result if isinstance(msg.result, str) else ""
                result_text = (result_text + "\n\n" + result).strip() if outcome is not None else result
                if msg.is_error and not aborted:
                    err = result or "; ".join(msg.errors or []) or msg.subtype
                    outcome = TurnOutcome(False, result_text, error=str(err)[:300], session_id=self.session_id)
                else:
                    outcome = TurnOutcome(not aborted, result_text, interrupted=aborted, session_id=self.session_id)
                if aborted or self._aborting:
                    return await self._abort(outcome, pending)

    async def _abort(self, outcome: TurnOutcome | None, pending: set[str]) -> TurnOutcome:
        """Durdur: give Claude a moment to end the turn, else close the CLI (kills its background shells)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.abort_grace
        from claude_agent_sdk import ResultMessage

        while outcome is None and loop.time() < deadline:
            try:
                msg = await asyncio.wait_for(self._inbox.get(), deadline - loop.time())
            except TimeoutError:
                break
            if msg is _EOF or isinstance(msg, Exception):
                break
            if isinstance(msg, ResultMessage):
                outcome = TurnOutcome(False, interrupted=True, session_id=msg.session_id or self.session_id)
        if outcome is None or pending:
            # The turn did not end, or background shells would keep running: close the CLI.
            await self.close()
        return TurnOutcome(False, (outcome.text if outcome else ""), interrupted=True, session_id=self.session_id)

    async def interrupt(self) -> None:
        """Called from another task (Durdur): ask the CLI to stop and wake :meth:`turn`."""
        self._aborting = True
        client = self._client
        if client is not None:
            try:
                await asyncio.wait_for(client.interrupt(), 3)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"claude interrupt: {e}")
        self._inbox.put(_ABORT)

    async def close(self) -> None:
        client, self._client = self._client, None
        reader, self._reader = self._reader, None
        if client is not None:
            try:
                await client.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"claude disconnect: {e}")
        if reader is not None and not reader.done():
            reader.cancel()
        self._inbox = _Inbox()


class HermesBrain(Brain):
    """One Hermes request per message (the conversation lives on the server)."""

    kind = "hermes"
    stoppable = False  # Durdur only cuts the speech; the server keeps working

    def __init__(self, client, *, ensure_connected: Callable[[], object] | None = None):
        self.client = client
        self._ensure = ensure_connected

    async def turn(self, text: str, emit: Emit) -> TurnOutcome:
        from celal.hermes import HermesError

        if self._ensure is not None:
            ok = self._ensure()
            if asyncio.iscoroutine(ok):
                ok = await ok
            if ok is False:
                return TurnOutcome(False, error="Sunucuya bağlı değil; mesaj Hermes'e gönderilmedi")
        parts: list[str] = []
        try:
            async for ev in self.client.stream(text):
                if ev.kind == "tool":
                    emit("progress", f"Hermes araç kullanıyor: {ev.value}")
                    continue
                parts.append(ev.value)
                emit("text", ev.value)
        except HermesError as e:
            return TurnOutcome(False, "".join(parts), error=str(e))
        return TurnOutcome(True, "".join(parts))


def make_brain_factory(settings, hermes_client, ensure_connected=None, env: Mapping[str, str] | None = None):
    """``factory(tab) -> Brain`` used by the tab manager."""

    def factory(tab) -> Brain:
        if tab.kind == "hermes":
            return HermesBrain(hermes_client, ensure_connected=ensure_connected)
        return MacClaudeBrain(
            cwd=tab.cwd or settings.mac_cwd,
            model=settings.mac_model,
            claude=settings.mac_claude,
            session_id=tab.session_id,
        )

    return factory
