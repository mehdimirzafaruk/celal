"""Two exact voice commands gate the microphone, Hey-Siri style.

While the mic is open Celal starts *passive*: Whisper still runs locally, but
every transcript is dropped unless it is exactly ``"Celal dinlemeye başla"``.
That phrase switches to *active*: transcripts flow on to the Mac router and
to Hermes as before. ``"Celal dinlemeyi durdur"`` switches back to passive;
neither phrase is ever forwarded as a user message.

Matching is exact on words. Only case (Turkish İ/I aware), punctuation at
word edges (``"Celal, dinlemeye başla."``), surrounding whitespace and the
circumflex spelling of the name are normalised; partial matches, extra
words or similar sentences never trigger.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

START_PHRASE = "Celal dinlemeye başla"
STOP_PHRASE = "Celal dinlemeyi durdur"

START = "start"
STOP = "stop"
DROPPED = "dropped"

_TR_CASE = str.maketrans({"I": "ı", "İ": "i", "Â": "a", "â": "a", "Î": "i", "î": "i", "Û": "u", "û": "u"})


def _strip_edge_punctuation(word: str) -> str:
    start, end = 0, len(word)
    while start < end and unicodedata.category(word[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(word[end - 1]).startswith("P"):
        end -= 1
    return word[start:end]


def normalize(text: str) -> str:
    """Lower-case (Turkish aware), drop punctuation at word edges, collapse whitespace."""
    words = (_strip_edge_punctuation(w) for w in text.translate(_TR_CASE).lower().split())
    return " ".join(w for w in words if w)


_START_KEY = normalize(START_PHRASE)
_STOP_KEY = normalize(STOP_PHRASE)


def classify(text: str) -> str | None:
    """``START`` / ``STOP`` when the text is exactly one of the two commands, else None."""
    key = normalize(text)
    if key == _START_KEY:
        return START
    if key == _STOP_KEY:
        return STOP
    return None


class ListenState:
    """Passive/active flag shared by the pipeline gate and the core."""

    def __init__(self, active: bool = False, on_change: Callable[[bool], None] | None = None):
        self.active = active
        self.on_change = on_change
        self.dropped: list[str] = []
        self.commands: list[str] = []

    def set_active(self, active: bool) -> None:
        """Change the state without a notification (mic button, startup)."""
        self.active = bool(active)

    def _switch(self, active: bool) -> None:
        if self.active == active:
            return
        self.active = active
        if self.on_change is not None:
            self.on_change(active)

    def handle(self, text: str) -> str | None:
        """What to do with a final transcript: ``START``/``STOP`` (consumed), ``DROPPED``, or None (pass on)."""
        command = classify(text)
        if command == START:
            self.commands.append(START)
            self._switch(True)
            return START
        if command == STOP and self.active:
            self.commands.append(STOP)
            self._switch(False)
            return STOP
        if not self.active:
            self.dropped.append(text)
            return DROPPED
        return None


class VoiceCommandGate(FrameProcessor):
    """Place after the hallucination filter: consumes the two commands, drops speech while passive."""

    def __init__(self, state: ListenState, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            if self.state.handle(frame.text) is not None:
                return
        elif isinstance(frame, InterimTranscriptionFrame) and not self.state.active:
            self.state.dropped.append(frame.text)
            return
        await self.push_frame(frame, direction)
