"""Speaker echo gate: ignore the microphone while Celal is talking.

Without headphones Celal hears its own voice from the speakers; VAD then
starts a user turn, interrupts the reply and Celal never finishes. By default
(no ``--barge-in``) the mic input is dropped while the bot speaks and for a
short tail after the last audio: no VAD, no STT, no interruption.

``BotStartedSpeakingFrame`` / ``BotStoppedSpeakingFrame`` are pushed upstream
by the output transport, so a gate placed right after the input transport
sees them. A second gate after STT drops any transcription that arrives while
the gate is closed.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DEFAULT_TAIL_SECS = 0.4


class SpeakingWindow:
    """Shared state: closed while the bot speaks and ``tail`` seconds after."""

    def __init__(self, tail: float = DEFAULT_TAIL_SECS, clock: Callable[[], float] = time.monotonic):
        self.tail = tail
        self._clock = clock
        self.speaking = False
        self._open_at = 0.0
        self.dropped_audio = 0
        self.dropped_transcripts: list[str] = []

    def bot_started(self) -> None:
        self.speaking = True

    def bot_stopped(self) -> None:
        self.speaking = False
        self._open_at = self._clock() + self.tail

    @property
    def closed(self) -> bool:
        return self.speaking or self._clock() < self._open_at


class _WindowTracker(FrameProcessor):
    def __init__(self, window: SpeakingWindow, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    def _track(self, frame: Frame) -> None:
        if isinstance(frame, BotStartedSpeakingFrame):
            self.window.bot_started()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.window.bot_stopped()


class MicGate(_WindowTracker):
    """Drops mic audio while the speaking window is closed (place after the input transport)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self._track(frame)
        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM and self.window.closed:
            self.window.dropped_audio += 1
            return
        await self.push_frame(frame, direction)


class TranscriptGate(_WindowTracker):
    """Drops transcriptions that arrive while the window is closed (place after STT)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self._track(frame)
        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)) and self.window.closed:
            self.window.dropped_transcripts.append(frame.text)
            return
        await self.push_frame(frame, direction)
