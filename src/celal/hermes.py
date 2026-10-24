"""Client for the Hermes API server (OpenAI compatible, reached through the SSH tunnel).

Two modes:

- ``responses`` (default): ``POST /v1/responses`` with ``"conversation": name``;
  Hermes keeps the conversation on the server, so only the new message is sent.
- ``chat``: ``POST /v1/chat/completions`` with ``stream=true``; the history is
  kept here, client side.

Both stream Server-Sent Events. :meth:`HermesClient.stream` yields
``("text", delta)`` and ``("tool", name)`` events. The API key goes only into
the ``Authorization`` header; it is never logged or put into error messages.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from loguru import logger

from celal.config import Settings

VOICE_INSTRUCTIONS = (
    "Celal Mac uygulamasından sesli konuşuyorsun; cevabın sesli okunacak. "
    "Kısa konuş: çoğunlukla 1-3 cümle. Markdown, liste, tablo, emoji ve bağlantı kullanma. "
    "Uzun bir iş gerekiyorsa kısaca ne yapacağını söyle."
)


class HermesError(RuntimeError):
    """Hermes could not answer (unreachable, auth, server error). Message is Turkish."""


@dataclass
class Event:
    kind: str  # "text" | "tool"
    value: str


class SSEParser:
    """Incremental SSE parser: feed lines, get (event, data) pairs on blank lines."""

    def __init__(self):
        self._event: str | None = None
        self._data: list[str] = []

    def feed(self, line: str) -> tuple[str | None, str] | None:
        if line == "":
            if not self._data:
                self._event = None
                return None
            out = (self._event, "\n".join(self._data))
            self._event, self._data = None, []
            return out
        if line.startswith("event:"):
            self._event = line[6:].strip()
        elif line.startswith("data:"):
            self._data.append(line[5:].lstrip())
        return None

    def flush(self) -> tuple[str | None, str] | None:
        return self.feed("")


class HermesClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        instructions: str | None = VOICE_INSTRUCTIONS,
        read_timeout: float = 1200.0,  # claude_task can work for many minutes between events
    ):
        self.settings = settings
        self.instructions = instructions
        self.conversation = settings.conversation
        self.history: list[dict] = []
        self._transport = transport
        self._read_timeout = read_timeout
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                transport=self._transport,
                timeout=httpx.Timeout(connect=5.0, read=self._read_timeout, write=10.0, pool=5.0),
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        key = self.settings.read_api_key()
        if not key:
            raise HermesError(f"API anahtarı yok: {self.settings.api_key_file} okunamadı")
        return {"Authorization": f"Bearer {key}", "Accept": "text/event-stream"}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def new_conversation(self) -> str:
        """Start a fresh conversation (new server-side name / empty history)."""
        self.conversation = f"{self.settings.conversation}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.history = []
        return self.conversation

    async def ping(self, timeout: float = 4.0) -> tuple[bool, str]:
        """(reachable and authorized, reason). Never raises."""
        try:
            headers = self._headers()
        except HermesError as e:
            return False, str(e)
        try:
            r = await self._http().get("/v1/models", headers=headers, timeout=timeout)
        except httpx.TimeoutException:
            return False, "sunucu cevap vermiyor (zaman aşımı)"
        except httpx.HTTPError as e:
            return False, f"sunucuya bağlanılamadı ({type(e).__name__})"
        if r.status_code == 401:
            return False, "API anahtarı reddedildi (401)"
        if r.status_code != 200:
            return False, f"sunucu hatası (HTTP {r.status_code})"
        return True, "bağlı"

    async def stream(self, text: str) -> AsyncIterator[Event]:
        if self.settings.api == "chat":
            async for ev in self._stream_chat(text):
                yield ev
        else:
            async for ev in self._stream_responses(text):
                yield ev

    async def _post_stream(self, path: str, body: dict) -> AsyncIterator[tuple[str | None, str]]:
        headers = self._headers()
        try:
            async with self._http().stream("POST", path, json=body, headers=headers) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode(errors="replace")[:200]
                    if r.status_code == 401:
                        raise HermesError("API anahtarı reddedildi (401)")
                    raise HermesError(f"sunucu hatası (HTTP {r.status_code}): {detail}")
                parser = SSEParser()
                async for line in r.aiter_lines():
                    ev = parser.feed(line.rstrip("\r"))
                    if ev:
                        yield ev
                ev = parser.flush()
                if ev:
                    yield ev
        except httpx.TimeoutException as e:
            raise HermesError("sunucu zamanında cevap vermedi") from e
        except (httpx.RemoteProtocolError, httpx.ReadError) as e:
            raise HermesError("bağlantı cevap gelirken koptu (tünel yeniden kuruluyor olabilir)") from e
        except httpx.HTTPError as e:
            raise HermesError(f"sunucuya bağlanılamadı ({type(e).__name__})") from e

    async def _stream_responses(self, text: str) -> AsyncIterator[Event]:
        body: dict = {
            "model": self.settings.model,
            "input": text,
            "conversation": self.conversation,
            "stream": True,
        }
        if self.instructions:
            body["instructions"] = self.instructions
        async for event, data in self._post_stream("/v1/responses", body):
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                logger.debug(f"hermes: non-JSON SSE data ignored ({len(data)} bytes)")
                continue
            kind = d.get("type") or event
            if kind == "response.output_text.delta":
                delta = d.get("delta") or ""
                if delta:
                    yield Event("text", delta)
            elif kind == "response.output_item.added":
                item = d.get("item") or {}
                if item.get("type") == "function_call" and item.get("name"):
                    yield Event("tool", item["name"])
            elif kind in ("response.failed", "error", "response.error"):
                err = d.get("error") or (d.get("response") or {}).get("error") or {}
                msg = err.get("message") if isinstance(err, dict) else str(err)
                raise HermesError(f"Hermes hata verdi: {msg or kind}")
            elif kind == "response.completed":
                break

    async def _stream_chat(self, text: str) -> AsyncIterator[Event]:
        messages = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        user = {"role": "user", "content": text}
        messages += self.history + [user]
        body = {"model": self.settings.model, "messages": messages, "stream": True}
        reply: list[str] = []
        try:
            async for _event, data in self._post_stream("/v1/chat/completions", body):
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if d.get("error"):
                    err = d["error"]
                    raise HermesError(f"Hermes hata verdi: {err.get('message') if isinstance(err, dict) else err}")
                for choice in d.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        reply.append(delta)
                        yield Event("text", delta)
        finally:
            if reply:
                self.history += [user, {"role": "assistant", "content": "".join(reply)}]
