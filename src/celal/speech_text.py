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
