""""Mac'te yap": run a task with the Mac's own Claude Code (``claude -p``).

A request is routed here instead of Hermes when it starts with one of the
configured prefixes ("Mac'te …", "bilgisayarda …"), when the window's
"Mac'te çalıştır" toggle is on, or when Hermes calls the local Mac API
(:mod:`celal.mac_api`). The task runs as::

    claude -p "<task>" --output-format stream-json --verbose --dangerously-skip-permissions

in the configured working directory (default: home; "Mac'te ~/Kod/x içinde …"
overrides it). The user's ``~/.claude`` skills, hooks and login (Max
subscription) apply as usual. Tool-use events become short Turkish progress
lines; the final result is kept in full and a 1-2 sentence spoken summary is
made locally (no second model call).

Up to ``max_parallel`` tasks (``[mac] max_parallel``, default 3) run at the
same time, each with its own id, process, progress and result. Each task
declares ``access``: ``read`` (screenshots, listing, questions) or ``write``
(the default: it may change files). Two ``write`` tasks in the same working
directory are refused (:class:`MacConflict`) so two Claudes never edit the same
tree at once; readers never conflict. :meth:`MacTaskRunner.stop` kills one
task's whole process group (claude and whatever it started), or all of them.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from celal.speech_text import clean_for_speech

BUSY = "Mac'te aynı anda çalışan iş sınırı doldu"
CONFLICT = "aynı klasörde değişiklik yapan bir Mac işi zaten çalışıyor"
DEFAULT_MAX_PARALLEL = 3
READ = "read"
WRITE = "write"
ACCESS = (READ, WRITE)
APPEND_PROMPT = (
    "Bu görev Celal sesli asistanından geldi (Mac'te yap). Sonucu düz Türkçe yaz; "
    "ilk iki cümle sesli okunacak, en önemli bilgiyi (sonuç, sayı, hata) başa koy."
)
STREAM_LIMIT = 32 * 1024 * 1024  # one stream-json line can carry a large tool result

RUNNING = "running"
DONE = "done"
FAILED = "failed"
STOPPED = "stopped"


class MacBusy(RuntimeError):
    """No free slot: ``max_parallel`` Mac tasks are already running."""


class MacConflict(MacBusy):
    """A ``write`` task is already running in the same working directory."""

    def __init__(self, task: "MacTask"):
        super().__init__(CONFLICT)
        self.task = task


def same_dir(a: str, b: str) -> bool:
    """True when two paths name the same directory (symlinks, ``..`` and trailing slashes resolved)."""
    return os.path.realpath(os.path.expanduser(a)) == os.path.realpath(os.path.expanduser(b))


# ---- parsing ---------------------------------------------------------------

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
_PATH_HINT = re.compile(
    r"^(?P<path>~?/\S+?)[,:]?\s+(?:içinde|içindeki|içinden|klasöründe|klasöründeki|dizininde|dizinindeki|"
    r"reposunda|reposundaki|projesinde|projesindeki)\b",
    re.IGNORECASE,
)


def match_prefix(text: str, prefixes: Sequence[str]) -> str | None:
    """Text after a "Mac'te"-style prefix, or None when the text has no prefix."""
    norm = text.strip().translate(_APOSTROPHES)
    low = norm.casefold()
    for p in sorted(prefixes, key=len, reverse=True):
        q = p.strip().translate(_APOSTROPHES).casefold()
        if not q or not low.startswith(q):
            continue
        rest = norm[len(q):]
        if rest and not (rest[0].isspace() or rest[0] in ",:;-–—"):
            continue  # "Macteki", "bilgisayardaki" are not the prefix
        rest = rest.lstrip(" \t\n,:;-–—")
        return rest
    return None


def split_cwd(task: str, home: Path | None = None) -> tuple[str, str | None]:
    """("~/Kod/x içinde testleri çalıştır") → (task, expanded dir) when the dir exists.

    The task text is kept whole (the path is useful context for Claude).
    """
    m = _PATH_HINT.match(task)
    if not m:
        return task, None
    raw = m.group("path").rstrip(",.:;")
    home = home or Path.home()
    path = Path(home, raw[2:]) if raw.startswith("~/") else Path(raw)
    if raw == "~":
        path = home
    return task, (str(path) if path.is_dir() else None)


def parse_request(text: str, prefixes: Sequence[str], *, home: Path | None = None) -> tuple[str, str | None] | None:
    """(task, cwd or None) when the text is a Mac request, else None."""
    rest = match_prefix(text, prefixes)
    if rest is None:
        return None
    return split_cwd(rest, home)


# ---- progress lines --------------------------------------------------------


def _short(value: str, n: int = 70) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= n else value[: n - 1] + "…"


def _base(path: str) -> str:
    return Path(str(path)).name or str(path)


def describe_tool(name: str, inp: dict | None) -> str:
    """One short Turkish line for a Claude Code tool call."""
    inp = inp or {}
    if name == "Bash":
        return f"komut çalıştırıyor: {_short(inp.get('command', ''))}"
    if name == "Read":
        return f"dosya okuyor: {_base(inp.get('file_path', ''))}"
    if name in ("Edit", "MultiEdit", "NotebookEdit"):
        return f"dosya düzenliyor: {_base(inp.get('file_path') or inp.get('notebook_path', ''))}"
    if name == "Write":
        return f"dosya yazıyor: {_base(inp.get('file_path', ''))}"
    if name == "Grep":
        return f"arıyor: {_short(inp.get('pattern', ''), 50)}"
    if name == "Glob":
        return f"dosya arıyor: {_short(inp.get('pattern', ''), 50)}"
    if name == "WebFetch":
        return f"web sayfası okuyor: {_short(inp.get('url', ''), 60)}"
    if name == "WebSearch":
        return f"web'de arıyor: {_short(inp.get('query', ''), 60)}"
    if name in ("Task", "Agent"):
        return f"alt ajan çalıştırıyor: {_short(inp.get('description', ''), 50)}"
    if name == "TodoWrite":
        return "plan güncelliyor"
    if name == "Skill":
        return f"skill kullanıyor: {_short(inp.get('skill') or inp.get('name', ''), 40)}"
    if name.startswith("mcp__"):
        return f"araç kullanıyor: {name.split('__')[-1]}"
    return f"araç kullanıyor: {name}"


def spoken_summary(text: str, sentences: int = 2, max_chars: int = 280) -> str:
    """First sentences of the result, cleaned for TTS (no model call)."""
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("```") or set(line) <= set("-=*_|: "):
            continue
        lines.append(line)
    flat = clean_for_speech(" ".join(lines))
    flat = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", flat)  # markdown links
    flat = " ".join(flat.split())
    if not flat:
        return ""
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9\"'(])", flat)
    out = " ".join(parts[:sentences]).strip()
    if len(out) > max_chars:
        cut = out[:max_chars].rsplit(" ", 1)[0]
        out = cut.rstrip(",;:") + "…"
    return out


# ---- child process ---------------------------------------------------------

_KEEP_CLAUDE_ENV = {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"}


def child_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Celal's environment minus Claude-session / API-billing variables.

    When Celal itself was started from inside a Claude Code session, its
    ``CLAUDECODE`` / ``CLAUDE_CODE_*`` / ``ANTHROPIC_BASE_URL`` variables would
    leak into the child; an ``ANTHROPIC_API_KEY`` would bill the API instead of
    the Max login. ``~/.local/bin`` is added to PATH (where the installer puts claude).
    """
    env = dict(os.environ if env is None else env)
    out = {
        k: v
        for k, v in env.items()
        if k in _KEEP_CLAUDE_ENV or not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC_"))
    }
    local_bin = str(Path(out.get("HOME") or Path.home()) / ".local" / "bin")
    path = out.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    if local_bin not in path.split(":"):
        out["PATH"] = f"{path}:{local_bin}"
    return out


def resolve_claude(name: str, env: dict[str, str] | None = None) -> str | None:
    expanded = os.path.expanduser(name)
    if os.sep in expanded:
        return expanded if os.access(expanded, os.X_OK) else None
    return shutil.which(expanded, path=(env or child_env()).get("PATH"))


def claude_command(claude: str, task: str) -> list[str]:
    return [
        claude,
        "-p",
        task,
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--append-system-prompt",
        APPEND_PROMPT,
    ]


async def default_spawn(args: Sequence[str], cwd: str, env: dict[str, str]):
    return await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # own process group: Durdur kills claude and its children
        limit=STREAM_LIMIT,
    )


@dataclass
class MacTask:
    task_id: str
    task: str
    cwd: str
    source: str  # "celal" (window/terminal) | "hermes"
    access: str = WRITE  # "read" | "write": writers lock their cwd against other writers
    status: str = RUNNING
    progress: list[str] = field(default_factory=list)
    result: str = ""
    error: str = ""
    summary: str = ""
    started: float = field(default_factory=time.time)
    seconds: float | None = None
    num_turns: int | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _proc: object = field(default=None, repr=False)
    _worker: asyncio.Task | None = field(default=None, repr=False)
    _stopping: bool = field(default=False, repr=False)

    @property
    def finished(self) -> bool:
        return self.status != RUNNING

    async def wait(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
        except TimeoutError:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "task": self.task,
            "cwd": self.cwd,
            "source": self.source,
            "access": self.access,
            "progress": self.progress[-20:],
            "result": self.result,
            "error": self.error,
            "summary": self.summary,
            "seconds": self.seconds if self.seconds is not None else round(time.time() - self.started, 1),
            "num_turns": self.num_turns,
        }


class MacTaskRunner:
    """Runs up to ``max_parallel`` ``claude -p`` processes and reports what they do.

    ``on_event(kind, task)`` with kind ``start`` / ``progress`` (new line in
    ``task.progress``) / ``done`` is called on the event loop.
    """

    def __init__(
        self,
        *,
        claude: str = "claude",
        default_cwd: str = "~",
        timeout: float = 1800.0,
        on_event: Callable[[str, MacTask], None] | None = None,
        spawn=default_spawn,
        env: dict[str, str] | None = None,
        keep: int = 20,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ):
        self.claude = claude
        self.default_cwd = default_cwd
        self.timeout = timeout
        self._on_event = on_event
        self._spawn = spawn
        self._env = env
        self._keep = max(keep, max_parallel)
        self.max_parallel = max(1, int(max_parallel))
        self.tasks: dict[str, MacTask] = {}

    @property
    def running(self) -> list[MacTask]:
        """Unfinished tasks, oldest first."""
        return [t for t in self.tasks.values() if not t.finished]

    @property
    def current(self) -> MacTask | None:
        """The oldest running task (None when idle); kept for status displays."""
        running = self.running
        return running[0] if running else None

    @property
    def busy(self) -> bool:
        return any(not t.finished for t in self.tasks.values())

    def get(self, task_id: str) -> MacTask | None:
        return self.tasks.get(task_id)

    def _emit(self, kind: str, task: MacTask) -> None:
        if self._on_event:
            try:
                self._on_event(kind, task)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"mac on_event failed: {e}")

    def _progress(self, task: MacTask, line: str) -> None:
        if task.progress and task.progress[-1] == line:
            return
        task.progress.append(line)
        self._emit("progress", task)

    def conflict(self, cwd: str, access: str = WRITE) -> MacTask | None:
        """The running writer that blocks a ``write`` task in ``cwd``, if any."""
        if access != WRITE:
            return None
        for r in self.running:
            if r.access == WRITE and same_dir(r.cwd, cwd):
                return r
        return None

    def start(self, task: str, cwd: str | None = None, source: str = "celal", access: str = WRITE) -> MacTask:
        """Start a task in the background.

        Raises ``ValueError`` (bad input), :class:`MacBusy` (all ``max_parallel``
        slots taken) or :class:`MacConflict` (a writer already runs in ``cwd``).
        """
        task = (task or "").strip()
        if not task:
            raise ValueError("görev boş")
        if access not in ACCESS:
            raise ValueError(f"access 'read' ya da 'write' olmalı, '{access}' geldi")
        workdir = os.path.expanduser(cwd or self.default_cwd)
        if not os.path.isdir(workdir):
            raise ValueError(f"çalışma klasörü yok: {workdir}")
        if len(self.running) >= self.max_parallel:
            raise MacBusy(BUSY)
        blocker = self.conflict(workdir, access)
        if blocker is not None:
            raise MacConflict(blocker)
        t = MacTask(task_id=uuid.uuid4().hex[:10], task=task, cwd=workdir, source=source, access=access)
        self.tasks[t.task_id] = t
        for old_id in [k for k, v in self.tasks.items() if v.finished][: max(0, len(self.tasks) - self._keep)]:
            del self.tasks[old_id]
        self._emit("start", t)
        t._worker = asyncio.create_task(self._run(t), name=f"mac-task-{t.task_id}")
        return t

    async def _run(self, t: MacTask) -> None:
        env = child_env(self._env)
        claude = resolve_claude(self.claude, env)
        try:
            if claude is None:
                raise FileNotFoundError(f"claude komutu bulunamadı ({self.claude}); [mac] claude ayarına yolu yaz")
            proc = await self._spawn(claude_command(claude, t.task), t.cwd, env)
        except Exception as e:  # noqa: BLE001
            self._finish(t, FAILED, error=f"claude başlatılamadı: {e}")
            return
        t._proc = proc
        if t._stopping:  # Durdur came while claude was being started
            await _kill_group(proc)
        stderr_task = asyncio.create_task(_read_all(proc.stderr))
        got_result = False
        try:
            async with asyncio.timeout(self.timeout):
                async for raw in _lines(proc.stdout):
                    if self._handle_line(t, raw):
                        got_result = True
                        break  # claude exits right after the result event
            try:
                await asyncio.wait_for(proc.wait(), timeout=30)
            except TimeoutError:
                await _kill_group(proc)
        except TimeoutError:
            await _kill_group(proc)
            self._finish(t, FAILED, error=f"zaman aşımı: {int(self.timeout // 60)} dk doldu, görev durduruldu")
            return
        except asyncio.CancelledError:
            await _kill_group(proc)
            self._finish(t, STOPPED, error="durduruldu")
            raise
        except Exception as e:  # noqa: BLE001
            await _kill_group(proc)
            self._finish(t, FAILED, error=f"claude çıktısı okunamadı: {e}")
            return
        finally:
            t._proc = None
        err = await stderr_task
        if t._stopping:
            self._finish(t, STOPPED, error="durduruldu")
        elif t.finished:
            pass  # _handle_line already finished it (result event)
        elif got_result:
            self._finish(t, DONE)
        else:
            tail = " ".join(err.strip().splitlines()[-3:])[:300]
            self._finish(t, FAILED, error=f"claude sonuç vermeden çıktı (kod {proc.returncode}){': ' + tail if tail else ''}")

    def _handle_line(self, t: MacTask, raw: bytes) -> bool:
        line = raw.decode(errors="replace").strip()
        if not line:
            return False
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(f"mac: non-JSON line ignored ({len(line)} bytes)")
            return False
        kind = ev.get("type")
        if kind == "system" and ev.get("subtype") == "init":
            self._progress(t, f"Claude başladı ({ev.get('model', '?')}, {_short(t.cwd, 40)})")
        elif kind == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    self._progress(t, describe_tool(str(block.get("name", "")), block.get("input")))
        elif kind == "result":
            t.num_turns = ev.get("num_turns")
            text = ev.get("result")
            text = text if isinstance(text, str) else ""
            if ev.get("is_error") or ev.get("subtype") not in (None, "success"):
                self._finish(t, FAILED, result=text, error=_short(text or str(ev.get("subtype")), 300))
            else:
                self._finish(t, DONE, result=text)
            return True
        return False

    def _finish(self, t: MacTask, status: str, *, result: str | None = None, error: str = "") -> None:
        if t.finished:
            return
        t.status = status
        if result is not None:
            t.result = result
        t.error = error
        t.seconds = round(time.time() - t.started, 1)
        if status == DONE:
            t.summary = spoken_summary(t.result) or "Mac'teki iş bitti."
        elif status == STOPPED:
            t.summary = "Mac'teki işi durdurdum."
        else:
            t.summary = f"Mac'teki iş başarısız: {spoken_summary(error, 1, 160)}"
        t._done.set()
        self._emit("done", t)

    async def stop(self, task_id: str | None = None) -> bool:
        """Kill one running task (its whole process group), or all of them when ``task_id`` is None.

        True if at least one task was running.
        """
        targets = [t for t in self.running if task_id is None or t.task_id == task_id]
        if not targets:
            return False
        for t in targets:
            t._stopping = True
            if t._proc is not None:
                await _kill_group(t._proc)
            # else: _run kills it right after spawning (no cancel: it could orphan the child)
        await asyncio.gather(*(t.wait(5) for t in targets))
        for t in targets:
            if not t.finished:
                self._finish(t, STOPPED, error="durduruldu")
        return True

    async def shutdown(self) -> None:
        await self.stop()
        for t in self.tasks.values():
            if t._worker is not None and not t._worker.done():
                t._worker.cancel()


async def _lines(stream):
    """Yield lines; works for asyncio StreamReader and simple async iterables (tests)."""
    if stream is None:
        return
    if hasattr(stream, "readline"):
        while True:
            try:
                line = await stream.readline()
            except ValueError:  # line longer than the limit: skip the rest of it
                continue
            if not line:
                return
            yield line
    else:
        async for line in stream:
            yield line


async def _read_all(stream) -> str:
    if stream is None or not hasattr(stream, "read"):
        return ""
    try:
        data = await stream.read()
    except Exception:  # noqa: BLE001
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else str(data)


async def _kill_group(proc, grace: float = 3.0) -> None:
    if getattr(proc, "returncode", None) is not None:
        return
    pid = getattr(proc, "pid", None)

    def send(sig):
        try:
            if pid:
                os.killpg(pid, sig)
            elif sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError):
            pass

    send(signal.SIGTERM)
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=grace)
    except TimeoutError:
        send(signal.SIGKILL)
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=grace)
        except TimeoutError:
            pass
