"""Selectable voices: the list, where worker engines live, and the remembered choice.

Engines:

* ``ema``      EMA Lightning, in-process (small, fast, Turkish only)
* ``ema-tts``  EMA-TTS, the larger 65M sibling, in a worker process
* ``kitten-*`` KittenTTS 2 (1.7B, MLX), three Turkish speakers, one shared worker
* ``say``      macOS Yelda, no install

Worker engines live in their own venv under ``<models>/venvs/<worker>``
(``celal --install-voice kitten``); see :mod:`celal.voice_worker`. The choice
is remembered in ``<state_dir>/voice.json``. This module stays light (no
Pipecat import) so the CLI can read it before the model disk is configured;
the TTS service itself is :mod:`celal.tts_switch`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from celal.paths import models_dir

DEFAULT_VOICE = "ema"


@dataclass(frozen=True)
class Voice:
    id: str
    label: str
    engine: str  # "ema" | "say" | a worker name
    speaker: str | None = None  # speaker inside a multi-voice worker


VOICES: tuple[Voice, ...] = (
    Voice("ema", "EMA Lightning", "ema"),
    Voice("ema-tts", "EMA-TTS (büyük)", "ema-tts"),
    Voice("kitten-haber", "Kitten · Haber spikeri", "kitten", "TurkishNewsAnchor"),
    Voice("kitten-balikci", "Kitten · Balıkçı", "kitten", "TurkishFisherman"),
    Voice("kitten-babaanne", "Kitten · Babaanne", "kitten", "TurkishGrandmaCooking"),
    Voice("say", "Mac · Yelda", "say"),
)
BY_ID = {v.id: v for v in VOICES}

# What each worker venv needs (installed with uv, Python 3.12).
WORKER_PACKAGES: dict[str, list[str]] = {
    "kitten": ["kittenml[mlx]>=0.9.6"],
    "ema-tts": ["torch>=2.5", "numpy", "scipy", "soundfile", "safetensors", "huggingface_hub",
                "voxcpm", "pyloudnorm"],
}


# ---- worker venvs ---------------------------------------------------------
def venv_dir(worker: str, env: dict | None = None) -> Path:
    return models_dir(env) / "venvs" / worker


def worker_python(worker: str, env: dict | None = None) -> Path:
    return venv_dir(worker, env) / "bin" / "python"


def is_available(voice: Voice, disk: bool = True) -> bool:
    if voice.engine == "say":
        return True
    if not disk:
        return False
    if voice.engine == "ema":
        return True
    return worker_python(voice.engine).exists()


def install_worker(worker: str, printer: Callable[[str], None] = print) -> int:
    """Create ``<models>/venvs/<worker>`` and install its packages with uv."""
    if worker not in WORKER_PACKAGES:
        printer(f"bilinmeyen ses motoru: {worker} (seçenekler: {', '.join(WORKER_PACKAGES)})")
        return 2
    uv = shutil.which("uv") or str(Path("~/.local/bin/uv").expanduser())
    if not Path(uv).exists():
        printer("uv bulunamadı: https://docs.astral.sh/uv/ ile kur")
        return 2
    target = venv_dir(worker)
    target.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, UV_LINK_MODE="copy")  # the venv is on another disk than uv's cache
    steps = [
        [uv, "venv", "--python", "3.12", "--allow-existing", str(target)],
        [uv, "pip", "install", "--python", str(target / "bin" / "python"), *WORKER_PACKAGES[worker]],
    ]
    for cmd in steps:
        printer("$ " + " ".join(cmd))
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            printer(f"kurulum başarısız ({rc})")
            return rc
    printer(f"{worker} kuruldu: {target}")
    return 0


def load_choice(state_dir: str | os.PathLike) -> str | None:
    try:
        data = json.loads((Path(state_dir).expanduser() / "voice.json").read_text())
        return data.get("voice") if data.get("voice") in BY_ID else None
    except (OSError, ValueError):
        return None


def save_choice(state_dir: str | os.PathLike, voice_id: str) -> None:
    path = Path(state_dir).expanduser() / "voice.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"voice": voice_id}))
    except OSError as e:
        logger.warning(f"ses seçimi kaydedilemedi: {e}")
