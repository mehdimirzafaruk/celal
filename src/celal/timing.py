"""Per-turn latency: speech end → STT → first token → first audio.

Printed once per turn, when the first TTS audio of the reply appears::

    ⏱ stt=0.9s first_token=0.6s ttfa=0.3s total=1.8s

- ``stt``: end of user speech (VAD stop time minus its ``stop_secs`` window)
  until the final transcript.
- ``first_token``: transcript (or, in text mode, prompt submit) until the
  first ``LLMTextFrame``.
- ``ttfa``: first token until the first ``TTSAudioRawFrame``.
- ``total``: speech end (or submit) until the first audio.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


@dataclass
class TurnTiming:
    speech_end: float | None = None
    transcript: float | None = None
    first_token: float | None = None
    first_audio: float | None = None

    def line(self) -> str:
        def fmt(a: float | None, b: float | None) -> str:
            return f"{b - a:.2f}s" if a is not None and b is not None else "-"

        start = self.speech_end if self.speech_end is not None else self.transcript
        llm_start = self.transcript if self.transcript is not None else self.speech_end
        return (
            f"⏱ stt={fmt(self.speech_end, self.transcript)} "
            f"first_token={fmt(llm_start, self.first_token)} "
            f"ttfa={fmt(self.first_token, self.first_audio)} "
            f"total={fmt(start, self.first_audio)}"
        )


class TurnTimer:
    """Clock shared by the observer and the text-mode driver."""

    def __init__(self, emit: Callable[[str], None] = print, clock: Callable[[], float] = time.time):
        self._emit = emit
        self._clock = clock
        self.current = TurnTiming()
        self.history: list[TurnTiming] = []

    def now(self) -> float:
        return self._clock()

    def speech_ended(self, at: float | None = None) -> None:
        self.current = TurnTiming(speech_end=at if at is not None else self.now())

    def prompt_submitted(self) -> None:
        """Text mode: there is no speech; the submit moment is the STT end."""
        self.current = TurnTiming(transcript=self.now())

    def transcript(self) -> None:
        if self.current.first_token is not None:  # stale turn; start fresh
            self.current = TurnTiming()
        if self.current.transcript is None or self.current.speech_end is not None:
            self.current.transcript = self.now()

    def token(self) -> None:
        if self.current.first_token is None and (
            self.current.transcript is not None or self.current.speech_end is not None
        ):
            self.current.first_token = self.now()

    def audio(self) -> None:
        if self.current.first_audio is None and self.current.first_token is not None:
            self.current.first_audio = self.now()
            self.history.append(self.current)
            self._emit(self.current.line())

    def interrupted(self) -> None:
        """A reply cut off before its first audio is reported and dropped.

        Pipecat also broadcasts an interruption at the start of every user
        turn; with no reply in flight that must not lose the turn's marks.
        """
        if self.current.first_token is not None and self.current.first_audio is None:
            self._emit(self.current.line() + " (kesildi)")
            self.current = TurnTiming()


class TurnTimingObserver(BaseObserver):
    """Feeds a :class:`TurnTimer` from frames as they are first pushed."""

    def __init__(self, timer: TurnTimer, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self.timer = timer

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self.timer.speech_ended(frame.timestamp - (frame.stop_secs or 0.0))
        elif isinstance(frame, TranscriptionFrame):
            self.timer.transcript()
        elif isinstance(frame, LLMTextFrame):
            self.timer.token()
        elif isinstance(frame, TTSAudioRawFrame):
            self.timer.audio()
        elif isinstance(frame, InterruptionFrame):
            self.timer.interrupted()
