"""Local HTTP API so Hermes can run "Mac'te yap" tasks (Hermes → Mac).

Listens on ``127.0.0.1:<port>`` only (default 8643). The server reaches it
through the reverse SSH forward ``-R 127.0.0.1:8643:127.0.0.1:8643``. Every
request needs ``Authorization: Bearer <token>``; the token lives in
``~/.config/celal/mac-api-key`` (mode 600, generated on first run) and a copy
on the server. It is never logged.

::

    POST /task      {"task": "...", "cwd": "~/Kod/x"?}  → 202 {"task_id", "status"}
                                                       409 when a task is already running
    GET  /task/<id> → 200 {"task_id", "status", "progress", "result", "error", "summary", ...}
    GET  /health    → 200 {"ok": true, "busy": bool}

Deliberately tiny (HTTP/1.1, one request per connection, JSON only), so no
web framework is needed.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from celal.mac_task import BUSY, MacBusy, MacTaskRunner

MAX_BODY = 64 * 1024
REASONS = {200: "OK", 202: "Accepted", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
           405: "Method Not Allowed", 409: "Conflict", 413: "Payload Too Large", 500: "Internal Server Error"}


def ensure_token(path: str | Path) -> str:
    """Read the API token, creating it (mode 600, dir 700) on first run."""
    p = Path(path).expanduser()
    try:
        token = p.read_text().strip()
    except FileNotFoundError:
        token = ""
    if token:
        if p.stat().st_mode & 0o077:
            os.chmod(p, 0o600)
        return token
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    token = secrets.token_urlsafe(32)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    os.chmod(p, 0o600)
    logger.info(f"Mac API anahtarı oluşturuldu: {p}")
    return token


class MacApiServer:
    def __init__(
        self,
        runner: MacTaskRunner,
        token: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8643,
        on_request: Callable[[str, dict], None] | None = None,
    ):
        self.runner = runner
        self._token = token.encode()
        self.host = host
        self.port = port
        self._on_request = on_request
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        sock = self._server.sockets[0].getsockname() if self._server.sockets else None
        if sock:
            self.port = sock[1]
        logger.info(f"Mac API dinliyor: {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 2)
            except TimeoutError:
                pass
            self._server = None

    def _authorized(self, headers: dict[str, str]) -> bool:
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return False
        return hmac.compare_digest(auth[7:].strip().encode(), self._token)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, body = 500, {"error": "iç hata"}
        try:
            method, path, headers, raw = await asyncio.wait_for(_read_request(reader), timeout=10)
            status, body = self._route(method, path, headers, raw)
        except _HttpError as e:
            status, body = e.status, {"error": e.message}
        except TimeoutError:
            status, body = 400, {"error": "istek zamanında gelmedi"}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Mac API: {type(e).__name__}: {e}")
        data = json.dumps(body, ensure_ascii=False).encode()
        head = (
            f"HTTP/1.1 {status} {REASONS.get(status, '')}\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n"
        )
        try:
            writer.write(head.encode() + data)
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    def _route(self, method: str, path: str, headers: dict[str, str], raw: bytes) -> tuple[int, dict]:
        if not self._authorized(headers):
            return 401, {"error": "yetkisiz"}
        path = path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health" and method == "GET":
            return 200, {"ok": True, "busy": self.runner.busy}
        if path == "/task":
            if method != "POST":
                return 405, {"error": "POST /task bekleniyor"}
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return 400, {"error": "gövde JSON değil"}
            if not isinstance(payload, dict):
                return 400, {"error": "gövde JSON nesnesi olmalı"}
            task = str(payload.get("task") or "").strip()
            cwd = payload.get("cwd")
            if not task:
                return 400, {"error": "task boş"}
            try:
                t = self.runner.start(task, str(cwd) if cwd else None, source="hermes")
            except MacBusy:
                cur = self.runner.current
                return 409, {"error": BUSY, "running_task_id": cur.task_id if cur else None}
            except ValueError as e:
                return 400, {"error": str(e)}
            if self._on_request:
                self._on_request("task", {"task_id": t.task_id, "task": t.task, "cwd": t.cwd})
            return 202, {"task_id": t.task_id, "status": t.status, "cwd": t.cwd}
        if path.startswith("/task/"):
            if method != "GET":
                return 405, {"error": "GET /task/<id> bekleniyor"}
            t = self.runner.get(path[len("/task/"):])
            if t is None:
                return 404, {"error": "görev yok (Celal yeniden başladıysa eski görevler silinir)"}
            return 200, t.to_dict()
        return 404, {"error": "bilinmeyen yol"}


class _HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str], bytes]:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.LimitOverrunError as e:
        raise _HttpError(413, "başlık çok büyük") from e
    except asyncio.IncompleteReadError as e:
        raise _HttpError(400, "eksik istek") from e
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) < 2:
        raise _HttpError(400, "bozuk istek satırı")
    method, path = parts[0].upper(), parts[1]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    try:
        length = int(headers.get("content-length") or 0)
    except ValueError as e:
        raise _HttpError(400, "Content-Length bozuk") from e
    if length > MAX_BODY:
        raise _HttpError(413, "gövde çok büyük")
    body = await reader.readexactly(length) if length else b""
    return method, path, headers, body
