"""Pipecat pipeline assembly for Vera (voice and text modes)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSStoppedFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from jarvis.audio_io import SoundDeviceTransport, SoundDeviceTransportParams
from jarvis.llm_claude import ClaudeAgentLLMService
from jarvis.stt_filter import WhisperHallucinationFilter
from jarvis.timing import TurnTimer, TurnTimingObserver
from jarvis.tts_ema import EMALightningTTSService
from jarvis.tts_say import SayTTSService

IN_RATE = 16000
OUT_RATE = 24000

STT_MODELS = {
    "turbo": MLXModel.LARGE_V3_TURBO.value,
    "turbo-q4": MLXModel.LARGE_V3_TURBO_Q4.value,
}
LLM_MODELS = {"haiku": "haiku", "sonnet": "sonnet"}


@dataclass
class Config:
    tts: str = "ema"
    stt: str = "turbo"
    model: str = "haiku"
    min_words: int = 2
    play: bool = True
    record_path: str | None = None
    input_wav: str | None = None
    input_wav_tail_secs: float = 8.0
    printer: Callable[[str], None] = field(default=print)


def make_tts(cfg: Config) -> EMALightningTTSService | SayTTSService:
    on_sentence = lambda s: cfg.printer(f"[vera] {s}")  # noqa: E731
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


def make_llm(cfg: Config) -> ClaudeAgentLLMService:
    return ClaudeAgentLLMService(model=LLM_MODELS.get(cfg.model, cfg.model))


class ConsoleObserver(BaseObserver):
    """Prints ``[sen] …`` for final transcripts and surfaces pipeline errors."""

    def __init__(self, printer: Callable[[str], None] = print, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self._print = printer

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, TranscriptionFrame):
            self._print(f"[sen] {frame.text.strip()}")
        elif isinstance(frame, ErrorFrame):
            self._print(f"[hata] {frame.error}")


class ReplyTracker(BaseObserver):
    """Knows when a reply has been fully spoken (text mode waits on this)."""

    def __init__(self, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self.reset()

    def reset(self) -> None:
        self.llm_started = False
        self.llm_ended = False
        self.tts_stopped = False
        self.bot_speaking = False
        self.last_event = time.monotonic()
        self.errors: list[str] = []

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, LLMFullResponseStartFrame):
            self.llm_started = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.llm_ended = True
        elif isinstance(frame, TTSStoppedFrame) and self.llm_ended:
            self.tts_stopped = True
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_speaking = False
        elif isinstance(frame, ErrorFrame):
            self.errors.append(frame.error)
        else:
            return
        self.last_event = time.monotonic()

    def done(self, settle: float = 0.6) -> bool:
        idle = time.monotonic() - self.last_event > settle
        if self.llm_ended and self.tts_stopped and not self.bot_speaking and idle:
            return True
        # LLM failed or produced nothing to say.
        return self.llm_ended and not self.bot_speaking and idle and bool(self.errors)


@dataclass
class Built:
    worker: PipelineWorker
    transport: SoundDeviceTransport
    llm: ClaudeAgentLLMService
    tts: EMALightningTTSService | SayTTSService
    timer: TurnTimer
    reply: ReplyTracker
    user_aggregator: object


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


def build_voice(cfg: Config) -> Built:
    """Mic → Silero VAD + Smart Turn v3 → Whisper (tr) → Claude → TTS → speaker."""
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
        )
    )
    stt = make_stt(cfg)
    llm = make_llm(cfg)
    tts = make_tts(cfg)
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams()),
            user_turn_strategies=UserTurnStrategies(
                # While Vera speaks, at least `min_words` words are needed to
                # interrupt ("hı hı" does not); otherwise one word starts a turn.
                start=[MinWordsUserTurnStartStrategy(min_words=cfg.min_words)],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())],
            ),
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            WhisperHallucinationFilter(),
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    timer = TurnTimer(emit=cfg.printer)
    reply = ReplyTracker()
    worker = _worker(pipeline, [TurnTimingObserver(timer), ConsoleObserver(cfg.printer), reply])
    logger.debug(f"voice pipeline: stt={STT_MODELS[cfg.stt]} llm={cfg.model} tts={cfg.tts}")
    return Built(worker, transport, llm, tts, timer, reply, aggregators.user())


def build_text(cfg: Config) -> Built:
    """Typed prompt → Claude → TTS → speaker (and/or WAV)."""
    transport = SoundDeviceTransport(
        SoundDeviceTransportParams(
            audio_out_enabled=True,
            audio_out_sample_rate=OUT_RATE,
            play=cfg.play,
            record_path=cfg.record_path,
        )
    )
    llm = make_llm(cfg)
    tts = make_tts(cfg)
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)
    pipeline = Pipeline([aggregators.user(), llm, tts, transport.output(), aggregators.assistant()])
    timer = TurnTimer(emit=cfg.printer)
    reply = ReplyTracker()
    worker = _worker(pipeline, [TurnTimingObserver(timer), ConsoleObserver(cfg.printer), reply])
    return Built(worker, transport, llm, tts, timer, reply, aggregators.user())
