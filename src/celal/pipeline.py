"""Pipecat pipeline assembly for Celal (voice and text modes).

The pipeline is only Celal's ears and mouth: mic → VAD/turn → Whisper →
voice commands → user turn → :class:`TurnSink` (hands the finished turn to
the tab router in :mod:`celal.core`) and ``TTSSpeakFrame`` → TTS → speaker.
Answers come from the tabs' brains (Mac Claude Code / Hermes), not from a
Pipecat LLM service.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import (
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from celal.echo_gate import MicGate, SpeakingWindow, TranscriptGate
from celal.audio_io import SoundDeviceTransport, SoundDeviceTransportParams
from celal.stt_filter import WhisperHallucinationFilter
from celal.tts_ema import EMALightningTTSService
from celal.tts_say import SayTTSService
from celal.wake import ListenState, VoiceCommandGate

IN_RATE = 16000
OUT_RATE = 24000

STT_MODELS = {
    "turbo": MLXModel.LARGE_V3_TURBO.value,
    "turbo-q4": MLXModel.LARGE_V3_TURBO_Q4.value,
}


@dataclass
class Config:
    tts: str = "ema"
    stt: str = "turbo"
    # 0: TTS stops as soon as VAD hears the user (needs headphones; no echo
    # cancellation). N >= 1: only after N transcribed words (speaker-friendly).
    min_words: int = 0
    # False (default): mic is ignored while Celal speaks (+ tail), so speakers
    # work without headphones. True: talking over Celal stops it (headphones).
    barge_in: bool = False
    echo_tail_secs: float = 0.4
    play: bool = True
    record_path: str | None = None
    input_wav: str | None = None
    input_wav_tail_secs: float = 8.0
    mic_on_start: bool = False  # mic stays closed until the user opens it
    # True: an open mic starts passive; only "Celal dinlemeye başla" makes Celal
    # listen, "Celal dinlemeyi durdur" makes it passive again (celal.wake).
    wake: bool = True
    printer: Callable[[str], None] = field(default=print)
    print_sentences: bool = False


def make_tts(cfg: Config) -> EMALightningTTSService | SayTTSService:
    on_sentence = (lambda s: cfg.printer(f"[ses] {s}")) if cfg.print_sentences else None  # noqa: E731
    if cfg.tts == "ema":
        return EMALightningTTSService(sample_rate=OUT_RATE, on_sentence=on_sentence)
    if cfg.tts == "say":
        return SayTTSService(sample_rate=OUT_RATE, on_sentence=on_sentence)
    raise ValueError(f"unknown tts: {cfg.tts}")


def make_stt(cfg: Config) -> WhisperSTTServiceMLX:
    return WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(model=STT_MODELS[cfg.stt], language=Language.TR),
        # Speech end -> final transcript, measured: ~0.6-1.1 s for a 3-5 s utterance on M2.
        ttfs_p99_latency=1.5,
    )


def turn_start_strategies(min_words: int):
    """Only matters with barge-in; without it the mic is gated while Celal speaks."""
    if min_words and min_words > 0:
        return [MinWordsUserTurnStartStrategy(min_words=min_words)]
    return [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]


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


class TurnSink(FrameProcessor):
    """End of the ears: a finished spoken user turn goes to ``route(text)``, never further down.

    Sits where an LLM service would be; the user aggregator (VAD + Smart Turn)
    decides when the turn is over and pushes the context.
    """

    def __init__(self, route: Callable[[str], None], **kwargs):
        super().__init__(**kwargs)
        self._route = route

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            text = last_user_text(frame.context)
            frame.context.set_messages([])  # the tabs keep the conversation; do not grow this one
            if text:
                try:
                    self._route(text)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"turn sink: {e}")
            return
        await self.push_frame(frame, direction)


@dataclass
class Built:
    worker: PipelineWorker
    transport: SoundDeviceTransport
    tts: EMALightningTTSService | SayTTSService
    user_aggregator: object | None = None
    window: SpeakingWindow | None = None
    listen: ListenState | None = None  # voice commands gate (None: always listening)


def _worker(pipeline: Pipeline, observers: list[BaseObserver]) -> PipelineWorker:
    return PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=IN_RATE,
            audio_out_sample_rate=OUT_RATE,
            enable_metrics=True,
        ),
        observers=observers,
        idle_timeout_secs=None,
        enable_rtvi=False,
    )


def build_voice(
    cfg: Config,
    route: Callable[[str], None],
    observers: list[BaseObserver] | None = None,
    listen: ListenState | None = None,
) -> Built:
    """Mic → Silero VAD + Smart Turn v3 → Whisper (tr) → [voice commands] → turn → ``route``; TTS → speaker."""
    transport = SoundDeviceTransport(
        SoundDeviceTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=IN_RATE,
            audio_out_sample_rate=OUT_RATE,
            play=cfg.play,
            record_path=cfg.record_path,
            input_wav=cfg.input_wav,
            input_wav_tail_secs=cfg.input_wav_tail_secs,
            capture_on_start=cfg.mic_on_start,
        )
    )
    stt = make_stt(cfg)
    tts = make_tts(cfg)
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams()),
            user_turn_strategies=UserTurnStrategies(
                start=turn_start_strategies(cfg.min_words),
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())],
            ),
        ),
    )
    window = None if cfg.barge_in else SpeakingWindow(tail=cfg.echo_tail_secs)
    gate_in = [MicGate(window)] if window else []
    gate_stt = [TranscriptGate(window)] if window else []
    pipeline = Pipeline(
        [
            transport.input(),
            *gate_in,
            stt,
            *gate_stt,
            WhisperHallucinationFilter(),
            *([VoiceCommandGate(listen)] if listen is not None else []),
            aggregators.user(),
            TurnSink(route),
            tts,
            transport.output(),
        ]
    )
    worker = _worker(pipeline, list(observers or []))
    logger.debug(
        f"voice pipeline: stt={STT_MODELS[cfg.stt]} tts={cfg.tts} barge_in={cfg.barge_in} wake={listen is not None}"
    )
    return Built(worker, transport, tts, aggregators.user(), window, listen)


def build_text(cfg: Config, observers: list[BaseObserver] | None = None) -> Built:
    """Speech only: ``TTSSpeakFrame`` → TTS → speaker (and/or WAV)."""
    transport = SoundDeviceTransport(
        SoundDeviceTransportParams(
            audio_out_enabled=True,
            audio_out_sample_rate=OUT_RATE,
            play=cfg.play,
            record_path=cfg.record_path,
        )
    )
    tts = make_tts(cfg)
    pipeline = Pipeline([tts, transport.output()])
    worker = _worker(pipeline, list(observers or []))
    return Built(worker, transport, tts)
