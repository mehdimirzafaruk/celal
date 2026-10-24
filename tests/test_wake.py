"""Hey-Siri style voice commands: only "Celal dinlemeye başla" / "Celal dinlemeyi durdur" switch listening."""

import pytest
from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.tests.utils import run_test

from celal.wake import START, START_PHRASE, STOP, STOP_PHRASE, ListenState, VoiceCommandGate, classify, normalize


def tf(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="t")


# ---- exact phrase matching --------------------------------------------------


def test_phrases_are_the_two_agreed_sentences():
    assert START_PHRASE == "Celal dinlemeye başla"
    assert STOP_PHRASE == "Celal dinlemeyi durdur"


@pytest.mark.parametrize(
    "text",
    [
        "Celal dinlemeye başla",
        "celal dinlemeye başla",
        "CELAL DİNLEMEYE BAŞLA",  # Turkish dotted İ → i
        "  Celal dinlemeye başla.  ",
        "Celal dinlemeye başla!",
        "Celal, dinlemeye başla.",  # Whisper's vocative comma
        "Celal  dinlemeye\tbaşla",  # inner whitespace
        "“Celal dinlemeye başla”",
        "Celâl dinlemeye başla",  # circumflex spelling of the name
    ],
)
def test_start_phrase_variants(text):
    assert classify(text) == START


@pytest.mark.parametrize(
    "text",
    ["Celal dinlemeyi durdur", "CELAL DİNLEMEYİ DURDUR", "celal dinlemeyi durdur...", " Celal, dinlemeyi durdur! "],
)
def test_stop_phrase_variants(text):
    assert classify(text) == STOP


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Celal",
        "dinlemeye başla",  # missing the name
        "Celal dinlemeye",  # partial
        "Celal dinlemeye başlasana",  # similar word
        "Celal dinlemeye başla lütfen",  # extra word
        "Hey Celal dinlemeye başla",  # extra word in front
        "Celal dinlemeye başla ve hava nasıl",  # embedded in a sentence
        "Celal dinlemeyi başlat",
        "CELAL DINLEMEYI DURDUR",  # ASCII I lower-cases to ı in Turkish: a different word
        "Celal dinlemeyi durdur mu",
        "Celal dinlemeye durdur",
        "Celal dinleme başla",
        "Jelal dinlemeye başla",
        "Celal'e dinlemeye başla",
        "Mac'te dinlemeye başla",
        "Hava nasıl?",
    ],
)
def test_similar_sentences_do_not_trigger(text):
    assert classify(text) is None


def test_normalize_keeps_inner_apostrophes_and_strips_edge_punctuation():
    assert normalize("  “Celal'e, SELAM!”  ") == "celal'e selam"
    assert normalize("DİNLE Iğdır") == "dinle ığdır"


# ---- state machine ----------------------------------------------------------


def test_passive_drops_everything_but_the_start_phrase():
    changes = []
    st = ListenState(on_change=changes.append)
    assert st.active is False
    assert st.handle("merhaba Celal") == "dropped"
    assert st.handle("Mac'te testleri çalıştır") == "dropped"
    assert st.handle("Celal dinlemeyi durdur") == "dropped"  # stop while passive: nothing to stop, not forwarded
    assert st.dropped == ["merhaba Celal", "Mac'te testleri çalıştır", "Celal dinlemeyi durdur"]
    assert changes == []
    assert st.handle("Celal dinlemeye başla.") == START
    assert st.active is True and changes == [True]


def test_active_forwards_speech_and_stop_phrase_returns_to_passive():
    changes = []
    st = ListenState(active=True, on_change=changes.append)
    assert st.handle("hava nasıl") is None  # pass on to the agent
    assert st.handle("Celal dinlemeye başla") == START  # already active: consumed, no change
    assert st.active is True and changes == []
    assert st.handle("Celal dinlemeyi durdur") == STOP
    assert st.active is False and changes == [False]
    assert st.handle("bu artık duyulmaz") == "dropped"
    assert st.commands == [START, STOP]


def test_set_active_resets_without_notifying():
    changes = []
    st = ListenState(active=True, on_change=changes.append)
    st.set_active(False)
    assert st.active is False and changes == []


# ---- pipeline gate ----------------------------------------------------------


async def test_gate_only_passes_transcripts_while_active():
    st = ListenState()
    frames = [
        tf("merhaba"),  # passive: dropped
        InterimTranscriptionFrame(text="mer", user_id="", timestamp="t"),  # passive: dropped
        tf("Celal, dinlemeye başla."),  # command: consumed
        tf("hava nasıl"),  # active: passes
        InterimTranscriptionFrame(text="ya", user_id="", timestamp="t"),  # active: passes
        tf("Celal dinlemeyi durdur"),  # command: consumed
        tf("bunu duyma"),  # passive again: dropped
    ]
    down, _ = await run_test(VoiceCommandGate(st), frames_to_send=frames)
    texts = [f.text for f in down if isinstance(f, (TranscriptionFrame, InterimTranscriptionFrame))]
    assert texts == ["hava nasıl", "ya"]
    assert st.dropped == ["merhaba", "mer", "bunu duyma"]
    assert st.commands == [START, STOP]
    assert st.active is False


async def test_gate_passes_non_transcription_frames_in_passive_mode():
    from pipecat.frames.frames import UserStartedSpeakingFrame

    st = ListenState()
    down, _ = await run_test(VoiceCommandGate(st), frames_to_send=[UserStartedSpeakingFrame()])
    assert any(isinstance(f, UserStartedSpeakingFrame) for f in down)
