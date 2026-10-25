"""Out-of-process TTS worker: one heavy engine in its own venv.

KittenTTS 2 and EMA-TTS pull in large dependency trees (transformers,
diffusers, voxcpm, gradio ...) that must not share Celal's venv with Pipecat.
Each runs in ``<models>/venvs/<engine>`` and Celal talks to it over pipes.

This file imports nothing from :mod:`celal`; it is run directly by the
engine's own Python: ``<venv>/bin/python voice_worker.py <engine>``.

Protocol (stdout is binary, everything printed goes to stderr):

* frame = 1 type byte + 4-byte little-endian length + payload
* ``R`` ready (JSON ``{"rate": int, ...}``) once the model is loaded
* request: one JSON line on stdin ``{"text": str, "voice": str | null, "rate": int}``
* reply: ``A`` frames (mono int16 PCM at the requested rate), then ``E``;
  or ``X`` (utf-8 error text) instead of ``E``
"""

from __future__ import annotations

import json
import os
import struct
import sys
import traceback

CHUNK_SECONDS = 0.5


def _frame(out, kind: bytes, payload: bytes = b"") -> None:
    out.write(kind + struct.pack("<I", len(payload)) + payload)
    out.flush()


def _to_int16(audio, src_rate: int, dst_rate: int) -> bytes:
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate != dst_rate:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src_rate, dst_rate)
        audio = resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


class Kitten:
    """KittenTTS 2 (1.7B, MLX on Apple silicon). Text arrives already normalized."""

    def __init__(self):
        from kittenml import KittenTTS

        try:
            self.m = KittenTTS("KittenML/kitten-tts-2", backend="mlx")
        except Exception as e:  # noqa: BLE001 - no MLX extra: plain torch on CPU/MPS
            print(f"kitten: mlx unavailable ({e}), default backend", file=sys.stderr)
            self.m = KittenTTS("KittenML/kitten-tts-2")
        self.rate = int(self.m.sample_rate)

    def say(self, text: str, voice: str | None):
        return self.m.generate(text, voice=voice or "TurkishNewsAnchor", normalize=False)


class EmaTTS:
    """EMA-TTS (65M flow matching + VoxCPM2 AudioVAE, 48 kHz), from its Hugging Face repo."""

    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download

        root = snapshot_download("canberkkkkkk/ema-tts", allow_patterns=["*.py", "*.json", "ckpt/*"])
        sys.path.insert(0, root)
        from inference import EmaTTS as Model  # the repo's own loader

        device = os.environ.get("CELAL_EMA_TTS_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
        try:
            self.m = Model.from_pretrained(os.path.join(root, "ckpt"), device=device)
            self.m.say("Merhaba.")
        except Exception as e:  # noqa: BLE001 - an op missing on MPS: CPU always works
            if device == "cpu":
                raise
            print(f"ema-tts: {device} failed ({e}), cpu", file=sys.stderr)
            self.m = Model.from_pretrained(os.path.join(root, "ckpt"), device="cpu")
        self.device = str(self.m.device)
        self.rate = int(self.m.sample_rate)

    def say(self, text: str, voice: str | None):
        return self.m.say(text, seed=7)


ENGINES = {"kitten": Kitten, "ema-tts": EmaTTS}


def main(argv: list[str]) -> int:
    out = sys.stdout.buffer
    sys.stdout = sys.stderr  # libraries print; only frames may reach the pipe
    name = argv[0] if argv else ""
    try:
        engine = ENGINES[name]()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _frame(out, b"X", f"{name} yüklenemedi: {e}".encode())
        return 1
    _frame(out, b"R", json.dumps({"rate": engine.rate, "device": getattr(engine, "device", None)}).encode())
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            rate = int(req.get("rate") or engine.rate)
            pcm = _to_int16(engine.say(str(req["text"]), req.get("voice")), engine.rate, rate)
            step = int(rate * CHUNK_SECONDS) * 2
            for i in range(0, len(pcm), step):
                _frame(out, b"A", pcm[i : i + step])
            _frame(out, b"E")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _frame(out, b"X", str(e).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
