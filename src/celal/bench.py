"""`celal --bench`: Whisper Turkish accuracy/speed, TTS TTFA/RTF, peak RSS."""

from __future__ import annotations

import asyncio
import re
import resource
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

BENCH_SENTENCES = [
    "Bugün hava çok güzel, akşam yürüyüşe çıkmayı düşünüyorum.",
    "Yarın sabah saat dokuzda ekiple kısa bir toplantımız var.",
    "Lütfen raporu cuma gününe kadar bana gönder.",
]
TTS_SENTENCE = "Merhaba Mirza, bugün seninle hangi işi bitirelim?"


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024  # macOS: bytes, Linux: KiB


def normalize_tr(text: str) -> list[str]:
    text = text.replace("I", "ı").replace("İ", "i").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance in words, reference word count)."""
    r, h = normalize_tr(ref), normalize_tr(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)], len(r)


def make_turkish_wav(path: Path, sentences: list[str], rate: int = 16000) -> float:
    subprocess.run(
        ["say", "-v", "Yelda", "-o", str(path), "--file-format=WAVE", f"--data-format=LEI16@{rate}", "--", " ".join(sentences)],
        check=True,
    )
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def wav_to_float(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def bench_whisper(repo: str, audio: np.ndarray) -> dict:
    """Same call WhisperSTTServiceMLX.run_stt makes (language tr, temperature 0)."""
    import mlx_whisper

    out = {}
    for label in ("cold", "warm"):
        t = time.perf_counter()
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, temperature=0.0, language="tr")
        out[label] = time.perf_counter() - t
    out["text"] = " ".join(s.get("text", "").strip() for s in result.get("segments", [])).strip()
    return out


def bench_ema() -> dict:
    from celal.tts_ema import EMALightningTTSService

    t = time.perf_counter()
    engine = EMALightningTTSService.load_engine("cpu")
    load = time.perf_counter() - t
    t = time.perf_counter()
    list(engine.stream("Merhaba.", sample_rate=24000, seed=7))
    warm = time.perf_counter() - t
    t = time.perf_counter()
    ttfa = None
    samples = 0
    for chunk in engine.stream(TTS_SENTENCE, sample_rate=24000, seed=7):
        if ttfa is None:
            ttfa = time.perf_counter() - t
        samples += len(chunk)
    total = time.perf_counter() - t
    dur = samples / 24000
    return {"load": load, "warmup": warm, "ttfa": ttfa, "total": total, "audio": dur, "rtf": total / dur}


def bench_say() -> dict:
    from celal.tts_say import say_to_pcm

    asyncio.run(say_to_pcm("Merhaba.", "Yelda", 24000))
    t = time.perf_counter()
    pcm = asyncio.run(say_to_pcm(TTS_SENTENCE, "Yelda", 24000))
    total = time.perf_counter() - t
    dur = len(pcm) / 2 / 24000
    # Not streaming: the first audio is available only once the sentence is rendered.
    return {"load": 0.0, "warmup": 0.0, "ttfa": total, "total": total, "audio": dur, "rtf": total / dur}


def run_bench(stt_models: list[str], skip_ema: bool = False) -> int:
    from celal.pipeline import STT_MODELS

    rows: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "tr3.wav"
        dur = make_turkish_wav(wav, BENCH_SENTENCES)
        audio = wav_to_float(wav)
        ref = " ".join(BENCH_SENTENCES)
        print(f"Referans ({dur:.1f}s, say -v Yelda): {ref}\n")
        for name in stt_models:
            r = bench_whisper(STT_MODELS[name], audio)
            from celal.stt_filter import strip_hallucinations

            raw_errs, n = word_errors(ref, r["text"])
            filtered = strip_hallucinations(r["text"])
            errs, _ = word_errors(ref, filtered)
            print(f"[{name}] ham transkript:   {r['text']}")
            print(f"[{name}] filtre sonrası:   {filtered}")
            rows.append(
                f"| STT whisper {name:<8} | cold {r['cold']:.2f}s, warm {r['warm']:.2f}s for {dur:.1f}s audio "
                f"(RTF {r['warm'] / dur:.2f}) | kelime hatası ham {raw_errs}/{n}, filtreli {errs}/{n} "
                f"(WER {100 * errs / n:.1f}%) | RSS {peak_rss_mb():.0f} MB |"
            )
    print()
    tts_results = {}
    if not skip_ema:
        try:
            tts_results["ema"] = bench_ema()
        except Exception as e:  # noqa: BLE001
            rows.append(f"| TTS ema            | BROKEN: {type(e).__name__}: {e} | | |")
    tts_results["say"] = bench_say()
    for name, r in tts_results.items():
        rows.append(
            f"| TTS {name:<14} | TTFA {r['ttfa']:.3f}s, synth {r['total']:.2f}s for {r['audio']:.2f}s audio "
            f"(RTF {r['rtf']:.3f}); load {r['load']:.1f}s, warmup {r['warmup']:.2f}s | | RSS {peak_rss_mb():.0f} MB |"
        )
    print("| Bileşen | Süre | Doğruluk | Tepe RSS (kümülatif) |")
    print("|---|---|---|---|")
    for row in rows:
        print(row)
    print(f"\nTepe RSS (süreç): {peak_rss_mb():.0f} MB")
    return 0
