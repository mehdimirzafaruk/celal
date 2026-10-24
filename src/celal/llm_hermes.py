"""Hermes as a Pipecat LLM service.

Frame contract (same as Pipecat's own LLM services)::

    LLMContextFrame -> LLMFullResponseStartFrame -> LLMTextFrame* -> LLMFullResponseEndFrame

Only the newest user message of the context is sent: in ``responses`` mode the
conversation lives on the server, in ``chat`` mode :class:`HermesClient` keeps
the history. On interruption Pipecat cancels the task that streams the reply,
which closes the HTTP stream.

``on_event(kind, value)`` reports ``start``, ``text`` (raw delta, for the UI),
``tool`` (tool name), ``error`` (Turkish message) and ``end``. Text pushed to
TTS is cleaned of markdown/emoji.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.errors import ErrorCategory

from celal.hermes import HermesClient, HermesError
from celal.speech_text import clean_for_speech


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


class HermesLLMService(LLMService):
    def __init__(
        self,
        client: HermesClient,
        *,
        on_event: Callable[[str, str], None] | None = None,
        **kwargs,
    ):
        settings = LLMSettings(
            model=client.settings.model,
            system_instruction=None,
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
        self.client = client
        self._on_event = on_event

    def _emit(self, kind: str, value: str = "") -> None:
        if self._on_event:
            try:
                self._on_event(kind, value)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"on_event failed: {e}")

    def can_generate_metrics(self) -> bool:
        return True

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
        self._emit("start", prompt)
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        first = True
        try:
            async for ev in self.client.stream(prompt):
                if ev.kind == "tool":
                    self._emit("tool", ev.value)
                    continue
                if first:
                    await self.stop_ttfb_metrics()
                    first = False
                self._emit("text", ev.value)
                spoken = clean_for_speech(ev.value)
                if spoken:
                    await self._push_llm_text(spoken)
        except asyncio.CancelledError:
            self._emit("end", "kesildi")
            raise
        except HermesError as e:
            self._emit("error", str(e))
            await self.push_error(error_msg=f"Hermes: {e}", category=ErrorCategory.CONNECTIVITY)
        except Exception as e:  # noqa: BLE001
            logger.exception("Hermes stream failed")
            self._emit("error", f"beklenmeyen hata: {e}")
            await self.push_error(error_msg=f"Hermes: {e}", category=ErrorCategory.UNKNOWN)
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
        self._emit("end", "")
        await self.push_frame(LLMFullResponseEndFrame())
