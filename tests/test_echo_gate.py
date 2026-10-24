"""Echo gate: mic audio and transcripts are dropped while Celal speaks (+ tail)."""

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from celal.echo_gate import MicGate, SpeakingWindow, TranscriptGate


def audio(tag: int) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=bytes([tag, 0]) * 160, sample_rate=16000, num_channels=1)


def test_window_closed_while_speaking_and_during_tail():
    t = [0.0]
    w = SpeakingWindow(tail=0.4, clock=lambda: t[0])
    assert not w.closed
    w.bot_started()
    t[0] = 5.0
    assert w.closed
    w.bot_stopped()
    t[0] = 5.39
    assert w.closed  # tail
    t[0] = 5.41
    assert not w.closed


async def test_mic_gate_drops_audio_while_speaking_and_passes_after_tail():
    window = SpeakingWindow(tail=0.2)
    frames = [
        audio(1),  # before: passes
        BotStartedSpeakingFrame(),
        audio(2),  # speaking: dropped
        audio(3),
        BotStoppedSpeakingFrame(),
        audio(4),  # inside the tail: dropped
        SleepFrame(0.35),
        audio(5),  # after the tail: passes
    ]
    down, _ = await run_test(MicGate(window), frames_to_send=frames)
    passed = [f.audio[0] for f in down if isinstance(f, InputAudioRawFrame)]
    assert passed == [1, 5]
    assert window.dropped_audio == 3
    # speaking frames themselves still flow on
    assert any(isinstance(f, BotStartedSpeakingFrame) for f in down)


async def test_transcript_gate_drops_text_heard_while_speaking():
    window = SpeakingWindow(tail=0.2)
    frames = [
        BotStartedSpeakingFrame(),
        TranscriptionFrame(text="Merhaba ben Celal", user_id="", timestamp="t"),
        BotStoppedSpeakingFrame(),
        SleepFrame(0.35),
        TranscriptionFrame(text="Sus artık", user_id="", timestamp="t"),
    ]
    down, _ = await run_test(TranscriptGate(window), frames_to_send=frames)
    texts = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    assert texts == ["Sus artık"]
    assert window.dropped_transcripts == ["Merhaba ben Celal"]
