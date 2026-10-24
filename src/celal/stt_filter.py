"""Drop well-known Whisper Turkish hallucinations (subtitle credits on silence).

Whisper was trained on subtitled video, so trailing silence or noise often
becomes "Altyazı M.K." or "İzlediğiniz için teşekkürler". These are removed
from final transcripts; a transcript that becomes empty is dropped.
"""

from __future__ import annotations

import re

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

HALLUCINATIONS = [
    r"altyaz[ıi]\s*m\.?\s*k\.?",
    r"altyaz[ıi]\s+\S+\s+taraf[ıi]ndan\s+(haz[ıi]rlanm[ıi][şs]t[ıi]r|yap[ıi]lm[ıi][şs]t[ıi]r)\.?",
    r"izledi[ğg]iniz\s+i[çc]in\s+te[şs]ekk[üu]r(ler|\s+ederim)\.?",
    r"abone\s+olmay[ıi]\s+unutmay[ıi]n\.?",
    r"bir\s+sonraki\s+videoda\s+g[öo]r[üu][şs]mek\s+[üu]zere\.?",
]
_PATTERN = re.compile(r"\s*(?:" + "|".join(HALLUCINATIONS) + r")\s*", re.IGNORECASE)


def strip_hallucinations(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _PATTERN.sub(" ", text)).strip()


class WhisperHallucinationFilter(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            cleaned = strip_hallucinations(frame.text)
            if not cleaned:
                return
            frame.text = cleaned
        await self.push_frame(frame, direction)
