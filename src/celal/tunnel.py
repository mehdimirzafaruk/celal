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


def _ssh_base(settings: Settings) -> list[str]:
    return [
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
    ] + (["-i", str(Path(settings.ssh_key).expanduser()), "-o", "IdentitiesOnly=yes"] if settings.ssh_key else [])


def ssh_command(settings: Settings) -> list[str]:
    """Mac → server: local port forwarded to Hermes on the server's loopback."""
    assert settings.ssh_host
    forward = ["-L", f"{settings.local_port}:127.0.0.1:{settings.remote_port}"]
    return ["ssh", "-N", *forward, *_ssh_base(settings), settings.ssh_host]


def reverse_ssh_command(settings: Settings) -> list[str]:
    """Server → Mac: the server's 127.0.0.1:<mac_api_port> reaches Celal's Mac API.

    A separate ssh process on purpose: if the server still holds the port from a
    dead session (Mac slept), only Hermes → Mac waits; Mac → Hermes keeps working.
    """
    assert settings.ssh_host
    port = settings.mac_api_port
    forward = ["-R", f"127.0.0.1:{port}:127.0.0.1:{port}"]
    return ["ssh", "-N", *forward, *_ssh_base(settings), settings.ssh_host]


def stale_listener_cleanup_command(settings: Settings) -> list[str]:
    """Kill an sshd session on the server that still listens on the Mac API port.

    Only ``sshd`` processes of the ssh user that listen on exactly that port are
    touched (the user cannot kill anything else anyway); a dead Mac session
    otherwise holds it until TCP keepalive gives up (~2 h).
    """
    assert settings.ssh_host
    port = int(settings.mac_api_port)
    remote = (
        f"ss -ltnpH 'sport = :{port}' 2>/dev/null | grep '\"sshd' "
        "| sed -n 's/.*pid=\\([0-9]*\\).*/\\1/p' | sort -u | xargs -r kill"
    )
    return ["ssh", *_ssh_base(settings), settings.ssh_host, remote]


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
        self.open_failed = False

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
                if self.open_failed:
                    self._set(DROPPED, f"ssh çalışıyor ama port {self.settings.local_port} {self._open_timeout:.0f} s'de açılmadı; yeniden deneniyor")
                else:
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
        opened = False
        self.open_failed = False
        while not wait_task.done() and time.monotonic() < deadline and not self._stop.is_set():
            if await self._port_open(self.settings.local_port):
                self._set(OPEN, f"127.0.0.1:{self.settings.local_port} → sunucu:{self.settings.remote_port}")
                opened = True
                break
            await asyncio.wait({wait_task}, timeout=0.2)
        if not opened and not wait_task.done() and not self._stop.is_set():
            # ssh is alive but the forward never came up (hung handshake, DNS,
            # half-open network): do not sit in "bağlanıyor" forever.
            self.open_failed = True
            await _terminate(proc)
            return await wait_task
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


class ReverseTunnel:
    """Keeps ``ssh -N -R 127.0.0.1:P:127.0.0.1:P`` alive (Hermes → Mac API).

    Considered open once ssh has stayed up for ``settle`` seconds (with
    ``ExitOnForwardFailure`` a refused remote bind exits at once). When the
    server says the port is taken, a stale session from a dead connection
    holds it: the cleanup command kills that session once, then ssh retries.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        spawn: Callable[[Sequence[str]], Awaitable[Proc]] = default_spawn,
        run_cleanup: Callable[[Sequence[str]], Awaitable[int]] | None = None,
        backoff: Sequence[float] = (2.0, 5.0, 10.0, 30.0, 60.0),
        settle: float = 3.0,
        stable_after: float = 60.0,
        on_status: Callable[[str, str], None] | None = None,
    ):
        self.settings = settings
        self._spawn = spawn
        self._run_cleanup = run_cleanup or _run_quiet
        self._backoff = list(backoff)
        self._settle = settle
        self._stable_after = stable_after
        self._on_status = on_status
        self._stop = asyncio.Event()
        self._proc: Proc | None = None
        self._task: asyncio.Task | None = None
        self.status = STOPPED
        self.detail = ""
        self.starts = 0
        self.cleanups = 0

    def _set(self, status: str, detail: str = "") -> None:
        if (status, detail) != (self.status, self.detail):
            logger.debug(f"reverse tunnel: {status} {detail}")
        self.status, self.detail = status, detail
        if self._on_status:
            self._on_status(status, detail)

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="ssh-reverse")
        return self._task

    async def _sleep(self, seconds: float) -> bool:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False

    async def run(self) -> None:
        try:
            await self._run()
        finally:
            self.kill_now()

    def kill_now(self) -> None:
        proc = self._proc
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def _run(self) -> None:
        failures = 0
        cleaned = False
        port = self.settings.mac_api_port
        while not self._stop.is_set():
            if not self.settings.ssh_host:
                self._set(NO_CONFIG, "ssh host ayarlı değil")
                if await self._sleep(10):
                    break
                continue
            self._set(CONNECTING)
            started = time.monotonic()
            try:
                proc = await self._spawn(reverse_ssh_command(self.settings))
            except OSError as e:
                self._set(DROPPED, f"ssh başlatılamadı: {e}")
                proc = None
            if proc is not None:
                self._proc = proc
                self.starts += 1
                wait_task = asyncio.ensure_future(proc.wait())
                done, _ = await asyncio.wait({wait_task}, timeout=self._settle)
                if not done and not self._stop.is_set():
                    self._set(OPEN, f"sunucu 127.0.0.1:{port} → Mac 127.0.0.1:{port}")
                    cleaned = False
                stop_task = asyncio.ensure_future(self._stop.wait())
                await asyncio.wait({wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
                stop_task.cancel()
                if not wait_task.done():
                    await _terminate(proc)
                rc = await wait_task
                self._proc = None
                if self._stop.is_set():
                    break
                err = await _stderr_tail(proc)
                if time.monotonic() - started >= self._stable_after:
                    failures = 0
                if "forwarding failed" in err.lower() or "remote port forwarding" in err.lower():
                    self._set(DROPPED, f"sunucu {port}'te dinlemeye izin vermedi (eski bağlantı portu tutuyor ya da anahtarın permitlisten izni yok)")
                    if not cleaned:
                        cleaned = True
                        self.cleanups += 1
                        try:
                            await self._run_cleanup(stale_listener_cleanup_command(self.settings))
                        except Exception as e:  # noqa: BLE001
                            logger.debug(f"reverse tunnel cleanup failed: {e}")
                        continue  # retry right away once
                else:
                    self._set(DROPPED, f"ssh çıktı (kod {rc}){': ' + err if err else ''}")
            delay = self._backoff[min(failures, len(self._backoff) - 1)]
            failures += 1
            if await self._sleep(delay):
                break
        self._set(STOPPED)

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


async def _run_quiet(args: Sequence[str], timeout: float = 20.0) -> int:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        return await asyncio.wait_for(proc.wait(), timeout)
    except TimeoutError:
        proc.kill()
        return -9


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
