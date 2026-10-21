from jarvis.timing import TurnTimer


def test_turn_timer_line():
    t = [100.0]
    lines = []
    timer = TurnTimer(emit=lines.append, clock=lambda: t[0])
    timer.speech_ended(at=100.0)
    t[0] = 100.9
    timer.transcript()
    timer.interrupted()  # turn-start interruption with no reply in flight: ignored
    t[0] = 101.5
    timer.token()
    t[0] = 101.8
    timer.audio()
    timer.audio()  # only first audio counts
    assert lines == ["⏱ stt=0.90s first_token=0.60s ttfa=0.30s total=1.80s"]


def test_text_mode_has_no_stt():
    t = [0.0]
    lines = []
    timer = TurnTimer(emit=lines.append, clock=lambda: t[0])
    timer.prompt_submitted()
    t[0] = 1.0
    timer.token()
    t[0] = 1.5
    timer.audio()
    assert lines == ["⏱ stt=- first_token=1.00s ttfa=0.50s total=1.50s"]


def test_hallucination_filter():
    from jarvis.stt_filter import strip_hallucinations

    assert strip_hallucinations("Lütfen raporu bana gönder. Altyazı M.K.") == "Lütfen raporu bana gönder."
    assert strip_hallucinations("İzlediğiniz için teşekkürler.") == ""
    assert strip_hallucinations("Merhaba Vera") == "Merhaba Vera"
