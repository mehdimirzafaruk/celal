"""SSH tunnel to the Hermes server, kept alive in the background.

``ssh -N -L <local>:127.0.0.1:<remote> host`` runs as a child process. When it
exits (network drop, sleep, server restart) it is started again with a short
backoff. If something else already listens on the local port (for example a
tunnel the user opened by hand) Celal uses it and does not start its own.
A health monitor can call :meth:`TunnelManager.kick` when the tunnel looks
alive but requests hang; the stuck ``ssh`` is killed and restarted.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from celal.config import Settings

# Tunnel states (shown in the UI).
NO_CONFIG = "ayar yok"
CONNECTING = "bağlanıyor"
OPEN = "açık"
EXTERNAL = "harici tünel"
DROPPED = "koptu"
STOPPED = "kapalı"


class Proc(Protocol):
    returncode: int | None

    async def wait(self) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


async def default_spawn(args: Sequence[str]) -> Proc:
    return await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # Ctrl-C in the terminal goes to Celal, which then stops ssh
    )


async def default_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (OSError, TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


def ssh_command(settings: Settings) -> list[str]:
    assert settings.ssh_host
    args = [
        "ssh",
        "-N",
        "-L",
        f"{settings.local_port}:127.0.0.1:{settings.remote_port}",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if settings.ssh_key:
        args += ["-i", str(Path(settings.ssh_key).expanduser()), "-o", "IdentitiesOnly=yes"]
    return args + [settings.ssh_host]


class TunnelManager:
    def __init__(
        self,
        settings: Settings,
        *,
        spawn: Callable[[Sequence[str]], Awaitable[Proc]] = default_spawn,
        port_open: Callable[[int], Awaitable[bool]] = default_port_open,
        backoff: Sequence[float] = (1.0, 2.0, 5.0, 10.0),
        open_timeout: float = 15.0,
        external_recheck: float = 5.0,
        stable_after: float = 30.0,
        on_status: Callable[[str, str], None] | None = None,
    ):
        self.settings = settings
        self._spawn = spawn
        self._port_open = port_open
        self._backoff = list(backoff)
        self._open_timeout = open_timeout
        self._external_recheck = external_recheck
        self._stable_after = stable_after
        self._on_status = on_status
        self._stop = asyncio.Event()
        self._proc: Proc | None = None
        self._task: asyncio.Task | None = None
        self.status = STOPPED
        self.detail = ""
        self.starts = 0

    def _set(self, status: str, detail: str = "") -> None:
        if (status, detail) != (self.status, self.detail):
            logger.debug(f"tunnel: {status} {detail}")
        self.status, self.detail = status, detail
        if self._on_status:
            self._on_status(status, detail)

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="ssh-tunnel")
        return self._task

    async def _sleep(self, seconds: float) -> bool:
        """Sleep unless stopped; returns True if stop was requested."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False

    async def run(self) -> None:
        try:
            await self._run()
        finally:
            # Cancelled (event loop teardown) while ssh runs: it is in its own
            # session, so nothing else would ever stop it.
            self.kill_now()

    def kill_now(self) -> None:
        """Synchronous last resort: SIGKILL our ssh if it is still running."""
        proc = self._proc
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            if await self._port_open(self.settings.local_port):
                self._set(EXTERNAL, f"127.0.0.1:{self.settings.local_port} zaten açık")
                if await self._sleep(self._external_recheck):
                    break
                continue
            if not self.settings.ssh_host:
                self._set(NO_CONFIG, "ssh host ayarlı değil (~/.config/celal/config.toml)")
                if await self._sleep(self._external_recheck):
                    break
                continue
            self._set(CONNECTING)
            started = time.monotonic()
            try:
                proc = await self._spawn(ssh_command(self.settings))
            except OSError as e:
                self._set(DROPPED, f"ssh başlatılamadı: {e}")
                proc = None
            if proc is not None:
                self._proc = proc
                self.starts += 1
                rc = await self._watch(proc)
                self._proc = None
                lived = time.monotonic() - started
                if self._stop.is_set():
                    break
                err = await _stderr_tail(proc)
                self._set(DROPPED, f"ssh çıktı (kod {rc}){': ' + err if err else ''}")
                if lived >= self._stable_after:
                    failures = 0
            delay = self._backoff[min(failures, len(self._backoff) - 1)]
            failures += 1
            if await self._sleep(delay):
                break
        self._set(STOPPED)

    async def _watch(self, proc: Proc) -> int:
        """Wait for the forward to come up, then for ssh to exit."""
        wait_task = asyncio.ensure_future(proc.wait())
        deadline = time.monotonic() + self._open_timeout
        while not wait_task.done() and time.monotonic() < deadline:
            if await self._port_open(self.settings.local_port):
                self._set(OPEN, f"127.0.0.1:{self.settings.local_port} → sunucu:{self.settings.remote_port}")
                break
            await asyncio.wait({wait_task}, timeout=0.2)
        stop_task = asyncio.ensure_future(self._stop.wait())
        await asyncio.wait({wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        stop_task.cancel()
        if not wait_task.done():
            await _terminate(proc)
        return await wait_task

    def kick(self, reason: str = "") -> bool:
        """Kill our ssh so it gets restarted (used when requests hang)."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return False
        logger.info(f"tunnel: restarting stuck ssh {reason}")
        try:
            proc.kill()
        except ProcessLookupError:
            return False
        return True

    async def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.returncode is None:
            await _terminate(proc)
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()


async def _terminate(proc: Proc, grace: float = 2.0) -> None:
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=grace)
    except TimeoutError:
        try:
            proc.kill()  # a SIGSTOPped ssh ignores SIGTERM until continued
        except ProcessLookupError:
            pass


async def _stderr_tail(proc: Any) -> str:
    stream = getattr(proc, "stderr", None)
    if stream is None:
        return ""
    try:
        data = await asyncio.wait_for(stream.read(), timeout=1)
    except (TimeoutError, Exception):  # noqa: BLE001
        return ""
    lines = [ln.strip() for ln in data.decode(errors="replace").splitlines() if ln.strip()]
    return lines[-1][:200] if lines else ""
