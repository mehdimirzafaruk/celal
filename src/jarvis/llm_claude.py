"""Claude Agent SDK as a Pipecat LLM service.

One ``ClaudeSDKClient`` lives for the whole session inside a dedicated asyncio
task (the SDK requires connect/query/disconnect in the same task context). The
multi-turn context is kept on the Claude side, so each Pipecat context frame
only contributes its latest user message.

Frame contract (same as Pipecat's own LLM services)::

    LLMContextFrame -> LLMFullResponseStartFrame -> LLMTextFrame* -> LLMFullResponseEndFrame

On ``InterruptionFrame`` Pipecat cancels the task that is streaming the reply;
the session then calls ``client.interrupt()`` and drains the aborted turn until
its ``ResultMessage`` arrives, so the next turn starts clean.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    StreamEvent,
    TextBlock,
)
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

VERA_SYSTEM_PROMPT = (
    "Adın Vera. Mirza'nın düşünme ortağısın; meslektaş gibi konuş, stajyer gibi değil. "
    "Sesli konuşuyorsun: kısa cümleler, en fazla 2–3 cümle, liste ve markdown yok, "
    "sayıları yazıyla söyle. Emin değilsen söyle, uydurma. Türkçe konuş; teknik "
    "terimler İngilizce kalabilir."
)

# Neutral working directory for the CLI process, so no project CLAUDE.md or
# .claude/ folder is picked up even by accident.
_NEUTRAL_CWD = Path.home() / ".cache" / "jarvis" / "claude-cwd"


def build_options(model: str, system_prompt: str = VERA_SYSTEM_PROMPT) -> ClaudeAgentOptions:
    """Options for a tool-less, settings-isolated, streaming chat session."""
    _NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
    return ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        tools=[],  # "[] (empty list) — Disable all built-in tools."
        allowed_tools=[],
        setting_sources=[],  # SDK isolation mode: no user/project/local settings, hooks, CLAUDE.md
        skills=[],  # hide every skill from the listing
        mcp_servers={},
        strict_mcp_config=True,  # ignore ~/.claude.json MCP servers
        include_partial_messages=True,
        thinking={"type": "disabled"},
        cwd=str(_NEUTRAL_CWD),
    )


def text_delta(event: dict[str, Any]) -> str | None:
    """Text of a raw Anthropic ``content_block_delta``/``text_delta`` stream event."""
    if event.get("type") != "content_block_delta":
        return None
    delta = event.get("delta") or {}
    if delta.get("type") != "text_delta":
        return None
    return delta.get("text") or None


def last_user_text(context: LLMContext) -> str | None:
    """The newest user message in a Pipecat context, flattened to plain text."""
    for message in reversed(context.get_messages()):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            text = " ".join(p for p in parts if p).strip()
            return text or None
    return None


# Items placed on a turn's output queue.
_TEXT = "text"
_END = "end"
_ERROR = "error"


@dataclass
class Turn:
    """One user prompt and the reply streaming back from Claude."""

    prompt: str
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    interrupt_sent: bool = False
    drained_messages: int = 0

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def emit(self, kind: str, value: Any = None) -> None:
        if self.cancelled:
            if kind != _END:
                self.drained_messages += 1
            return
        self.out.put_nowait((kind, value))


class ClaudeSession:
    """Owns a single ``ClaudeSDKClient`` inside one long-lived asyncio task."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        client_factory: Callable[..., Any] = ClaudeSDKClient,
        connect_timeout: float = 60.0,
    ):
        self._options = options
        self._client_factory = client_factory
        self._connect_timeout = connect_timeout
        self._requests: asyncio.Queue[Turn | None] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._connect_error: BaseException | None = None
        self._task: asyncio.Task | None = None
        self.current: Turn | None = None
        self.interrupts_sent = 0
        self.connect_seconds: float | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="claude-session")

    async def wait_ready(self) -> None:
        await asyncio.wait_for(self._ready.wait(), timeout=self._connect_timeout)
        if self._connect_error:
            raise RuntimeError(f"Claude SDK connect failed: {self._connect_error}") from self._connect_error

    def ask(self, prompt: str) -> Turn:
        turn = Turn(prompt=prompt)
        self._requests.put_nowait(turn)
        return turn

    def interrupt(self) -> bool:
        """Abort the turn in flight (if any). Returns True when something was cancelled."""
        turn = self.current
        hit = False
        if turn is not None and not turn.done.is_set() and not turn.cancelled:
            turn.cancel_event.set()
            hit = True
        # Turns queued but not yet sent are dropped as well.
        pending: list[Turn | None] = []
        while not self._requests.empty():
            pending.append(self._requests.get_nowait())
        for queued in pending:
            if queued is None:
                self._requests.put_nowait(None)
            else:
                queued.cancel_event.set()
                queued.done.set()
                hit = True
        return hit

    async def close(self, timeout: float = 5.0) -> None:
        if self._task is None:
            return
        self.interrupt()
        self._requests.put_nowait(None)
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Claude session closed with error: {e}")
        self._task = None

    async def _interrupt_when_cancelled(self, client: Any, turn: Turn) -> None:
        await turn.cancel_event.wait()
        turn.interrupt_sent = True
        self.interrupts_sent += 1
        logger.debug("Claude turn interrupted; sending client.interrupt() and draining")
        try:
            await client.interrupt()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"client.interrupt() failed: {e}")

    async def _run(self) -> None:
        client = self._client_factory(options=self._options)
        t0 = time.perf_counter()
        try:
            await client.connect()
        except BaseException as e:  # noqa: BLE001
            self._connect_error = e
            self._ready.set()
            logger.error(f"Claude SDK connect failed: {e}")
            return
        self.connect_seconds = time.perf_counter() - t0
        logger.debug(f"Claude SDK connected in {self.connect_seconds:.2f}s")
        self._ready.set()
        try:
            while True:
                turn = await self._requests.get()
                if turn is None:
                    break
                if turn.cancelled:
                    turn.done.set()
                    continue
                await self._run_turn(client, turn)
        finally:
            try:
                await client.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Claude SDK disconnect: {e}")

    async def _run_turn(self, client: Any, turn: Turn) -> None:
        self.current = turn
        watcher = asyncio.create_task(self._interrupt_when_cancelled(client, turn))
        got_delta = False
        try:
            await client.query(turn.prompt)
            async for msg in client.receive_response():
                if isinstance(msg, StreamEvent):
                    if msg.parent_tool_use_id is not None:
                        continue
                    text = text_delta(msg.event)
                    if text:
                        got_delta = True
                        turn.emit(_TEXT, text)
                elif isinstance(msg, AssistantMessage):
                    # Only used if the CLI did not stream partial messages.
                    if not got_delta:
                        for block in msg.content:
                            if isinstance(block, TextBlock) and block.text:
                                turn.emit(_TEXT, block.text)
                elif isinstance(msg, ResultMessage):
                    if msg.is_error and not turn.cancelled:
                        turn.emit(_ERROR, RuntimeError(f"Claude result error: {msg.subtype} {msg.errors or ''}"))
                    turn.emit(_END, msg)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Claude turn failed: {e}")
            turn.emit(_ERROR, e)
            turn.emit(_END, None)
        finally:
            watcher.cancel()
            self.current = None
            turn.done.set()
            if turn.cancelled:
                logger.debug(f"Drained interrupted turn ({turn.drained_messages} messages dropped)")


class ClaudeAgentLLMService(LLMService):
    """Pipecat ``LLMService`` backed by the Claude Agent SDK (subscription login, no API key)."""

    def __init__(
        self,
        *,
        model: str = "haiku",
        system_prompt: str = VERA_SYSTEM_PROMPT,
        session: ClaudeSession | None = None,
        on_text: Callable[[str], None] | None = None,
        **kwargs,
    ):
        settings = LLMSettings(
            model=model,
            system_instruction=system_prompt,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)
        self._session = session or ClaudeSession(build_options(model, system_prompt))
        self._on_text = on_text

    @property
    def session(self) -> ClaudeSession:
        return self._session

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session.start()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._session.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._session.close(timeout=2.0)

    async def warmup(self) -> None:
        """Start the CLI process early so the first turn does not pay for it."""
        self._session.start()
        await self._session.wait_ready()

    async def _handle_interruptions(self, frame: InterruptionFrame):
        await super()._handle_interruptions(frame)
        if self._session.interrupt():
            logger.debug(f"{self}: interruption -> Claude turn aborted")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self._process_context(frame.context)
        else:
            await self.push_frame(frame, direction)

    async def _process_context(self, context: LLMContext) -> None:
        prompt = last_user_text(context)
        if not prompt:
            logger.debug(f"{self}: context without a user message, skipping")
            return
        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        turn: Turn | None = None
        try:
            await self._session.wait_ready()
            turn = self._session.ask(prompt)
            first = True
            while True:
                kind, value = await turn.out.get()
                if kind == _TEXT:
                    if first:
                        await self.stop_ttfb_metrics()
                        first = False
                    if self._on_text:
                        self._on_text(value)
                    await self._push_llm_text(value)
                elif kind == _ERROR:
                    await self.push_error(error_msg=f"Claude: {value}", exception=value)
                elif kind == _END:
                    break
        except asyncio.CancelledError:
            # Interruption: Pipecat cancels this task. Make sure the SDK side stops too.
            if turn is not None and not turn.done.is_set():
                turn.cancel_event.set()
            raise
        except Exception as e:  # noqa: BLE001
            await self.push_error(error_msg=f"Claude: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
        await self.push_frame(LLMFullResponseEndFrame())
