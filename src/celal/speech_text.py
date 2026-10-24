"""Make streamed reply text speakable: drop markdown symbols and emoji.

Works on deltas (no state), so only character-level cleanup is done. The UI
shows the original text; only what goes to TTS is cleaned.
"""

from __future__ import annotations

import re

_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, symbols
    "☀-➿"  # misc symbols, dingbats
    "️‍"  # variation selector, zero-width joiner
    "]+"
)
_MARKUP = re.compile(r"[*`#~|>_]+")
_BULLET = re.compile(r"(^|\n)\s*(?:[-•]|\d+[.)])\s+")


def clean_for_speech(text: str) -> str:
    text = _BULLET.sub(r"\1", text)
    text = _EMOJI.sub("", text)
    text = _MARKUP.sub("", text)
    return text


_SENTENCE_END = re.compile(r"(?<=[.!?…:])\s+|\n+")


class SentenceBuffer:
    """Streamed reply text → whole sentences for TTS (code blocks and table rows skipped).

    ``feed(delta)`` returns the sentences completed by the delta; ``flush()``
    returns the rest. Sentences are cleaned with :func:`clean_for_speech`.
    """

    def __init__(self):
        self._buf = ""
        self._in_code = False

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            m = _SENTENCE_END.search(self._buf)
            if not m:
                break
            piece, self._buf = self._buf[: m.start()], self._buf[m.end():]
            out += self._emit(piece)
        return out

    def flush(self) -> list[str]:
        piece, self._buf = self._buf, ""
        out = self._emit(piece)
        self._in_code = False
        return out

    def _emit(self, piece: str) -> list[str]:
        line = piece.strip()
        if line.startswith("```"):
            self._in_code = not self._in_code
            return []
        if self._in_code or not line or line.startswith("|"):
            return []
        spoken = " ".join(clean_for_speech(line).split())
        return [spoken] if any(ch.isalnum() for ch in spoken) else []
