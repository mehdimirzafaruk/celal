"""Local HTTP API so Hermes can give the Mac work (Hermes → Mac), without waiting for it.

Listens on ``127.0.0.1:<port>`` only (default 8643). The server reaches it
through the reverse SSH forward ``-R 127.0.0.1:8643:127.0.0.1:8643``. Every
request needs ``Authorization: Bearer <token>``; the token lives in
``~/.config/celal/mac-api-key`` (mode 600, generated on first run) and a copy
on the server. It is never logged.

::

    POST /task              {"task": "...", "cwd"?: "~/Kod/x", "source"?: "telegram", "label"?: "..."}
                            → 202 {"task_id", "tab_id", "status": "running"|"queued", "queued": bool, "cwd"}
    GET  /task/<id>         → 200 {"task_id", "status": queued|running|done|error|cancelled, "result"?, "error"?,
                                   "started_at", "finished_at", "progress": [...], ...}
    POST /task/<id>/cancel  → 200 {"task_id", "cancelled": bool, "status"}
    GET  /tasks             → 200 {"tasks": [running / queued tabs], "running": n, "queued": n, "max_parallel": n}
    GET  /health            → 200 {"ok": true, "busy": bool, "running": n, "queued": n, "max_parallel": n}

Every task opens its own tab (📱) in the window and runs on the Mac's Claude
Code like a typed message. At most ``max_parallel`` tabs answer at once
(shared with the window); more wait as ``queued``. Results stay readable for
24 h, across Celal restarts (``~/.config/celal/tasks.json``).

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

from celal.tabs import MAC, QUEUED, RUNNING, TabManager, make_label

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
        tabs: TabManager,
        token: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8643,
        on_request: Callable[[str, dict], None] | None = None,
    ):
        self.tabs = tabs
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
        tabs = self.tabs
        if path == "/health" and method == "GET":
            return 200, {
                "ok": True,
                "busy": bool(tabs.running or tabs.queued),
                "running": len(tabs.running),
                "queued": len(tabs.queued),
                "max_parallel": tabs.max_parallel,
            }
        if path == "/tasks":
            if method != "GET":
                return 405, {"error": "GET /tasks bekleniyor"}
            return 200, self._tasks_view()
        if path == "/task":
            if method != "POST":
                return 405, {"error": "POST /task bekleniyor"}
            return self._start(raw)
        if path.startswith("/task/"):
            rest = path[len("/task/"):]
            if rest.endswith("/cancel"):
                if method != "POST":
                    return 405, {"error": "POST /task/<id>/cancel bekleniyor"}
                return self._cancel(rest[: -len("/cancel")])
            if method != "GET":
                return 405, {"error": "GET /task/<id> bekleniyor"}
            rec = tabs.tasks.get(rest)
            if rec is None:
                return 404, {"error": "görev yok (24 saatten eski ya da hiç olmadı)"}
            body = dict(rec)
            tab = tabs.get(rec.get("tab_id") or "")
            if tab is not None:
                body["progress"] = [ln["text"] for ln in tab.lines if ln.get("role") == "progress"][-20:]
            return 200, body
        return 404, {"error": "bilinmeyen yol"}

    def _start(self, raw: bytes) -> tuple[int, dict]:
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return 400, {"error": "gövde JSON değil"}
        if not isinstance(payload, dict):
            return 400, {"error": "gövde JSON nesnesi olmalı"}
        task = str(payload.get("task") or "").strip()
        if not task:
            return 400, {"error": "task boş"}
        cwd = payload.get("cwd")
        if cwd:
            cwd = os.path.expanduser(str(cwd))
            if not os.path.isdir(cwd):
                return 400, {"error": f"çalışma klasörü yok: {cwd}"}
        label = str(payload.get("label") or "").strip() or make_label(task)
        tabs = self.tabs
        queued = not tabs.slot_free()
        tab = tabs.create(MAC, source="hermes", label=label[:40], cwd=cwd or None, activate=False)
        rec = tabs.tasks.create(tab, task, queued=queued)
        if payload.get("source"):
            rec["origin"] = str(payload["source"])[:40]
        tabs.submit(tab, task, task_id=rec["task_id"])
        if self._on_request:
            self._on_request("task", {"task_id": rec["task_id"], "tab_id": tab.id, "task": task, "cwd": tab.cwd})
        return 202, {"task_id": rec["task_id"], "tab_id": tab.id, "status": QUEUED if queued else RUNNING,
                     "queued": queued, "cwd": tab.cwd, "label": tab.label}

    def _cancel(self, task_id: str) -> tuple[int, dict]:
        rec = self.tabs.tasks.get(task_id)
        if rec is None:
            return 404, {"error": "görev yok"}
        tab = self.tabs.get(rec.get("tab_id") or "")
        if rec.get("status") not in (QUEUED, RUNNING) or tab is None:
            return 200, {"task_id": task_id, "cancelled": False, "status": rec.get("status")}
        asyncio.ensure_future(self.tabs.stop(tab))
        return 200, {"task_id": task_id, "cancelled": True, "status": "cancelling"}

    def _tasks_view(self) -> dict:
        tabs = self.tabs
        by_tab = {r["tab_id"]: r for r in tabs.tasks.active()}
        items = []
        for t in tabs.tabs.values():
            if t.status not in (QUEUED, RUNNING):
                continue
            rec = by_tab.get(t.id)
            items.append({
                "task_id": rec["task_id"] if rec else None,
                "tab_id": t.id,
                "label": t.label,
                "status": t.status,
                "source": t.source,
                "kind": t.kind,
                "started_at": t.turn_started or None,
                "task": (rec or {}).get("task") or t.turn_text,
            })
        return {"tasks": items, "running": len(tabs.running), "queued": len(tabs.queued),
                "max_parallel": tabs.max_parallel}


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
