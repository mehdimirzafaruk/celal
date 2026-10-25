"""Selectable voices: the list, where worker engines live, and the remembered choice.

Engines:

* ``ema``      EMA Lightning, in-process (small, fast, Turkish only)
* ``ema-tts``  EMA-TTS, the larger 65M sibling, in a worker process
* ``kitten-*`` KittenTTS 2 (1.7B, MLX), three Turkish speakers, one shared worker
* ``say``      macOS Yelda, no install

Voice models and worker venvs live on the Mac's own disk, never on the
external model disk: ``CELAL_VOICES_DIR`` (default ``~/.local/share/celal/voices``),
so Celal speaks with the chosen voice even when that disk is unplugged.
Worker engines have their own venv under ``<voices>/venvs/<worker>``
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

DEFAULT_VOICE = "ema"
DEFAULT_VOICES_DIR = "~/.local/share/celal/voices"
MIN_FREE_GB = 15  # leave the Mac's disk this much room after an install


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


# ---- where voices live ----------------------------------------------------
def voices_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("CELAL_VOICES_DIR") or DEFAULT_VOICES_DIR).expanduser()


def hub_dir(env: Mapping[str, str] | None = None) -> Path:
    """Hugging Face cache for voice models (Whisper's stays on the model disk)."""
    return voices_dir(env) / "huggingface" / "hub"


def voices_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Cache variables for a worker process: every download goes under the voices dir."""
    root = voices_dir(env)
    return {
        "HF_HOME": str(root / "huggingface"),
        "HF_HUB_CACHE": str(hub_dir(env)),
        "HF_XET_CACHE": str(root / "huggingface" / "xet"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "TORCH_HOME": str(root / "torch"),
    }


def venv_dir(worker: str, env: Mapping[str, str] | None = None) -> Path:
    return voices_dir(env) / "venvs" / worker


def worker_python(worker: str, env: Mapping[str, str] | None = None) -> Path:
    return venv_dir(worker, env) / "bin" / "python"


def is_available(voice: Voice) -> bool:
    if voice.engine in ("say", "ema"):
        return True
    return worker_python(voice.engine).exists()


def install_worker(worker: str, printer: Callable[[str], None] = print) -> int:
    """Create ``<voices>/venvs/<worker>`` and install its packages with uv."""
    if worker not in WORKER_PACKAGES:
        printer(f"bilinmeyen ses motoru: {worker} (seçenekler: {', '.join(WORKER_PACKAGES)})")
        return 2
    uv = shutil.which("uv") or str(Path("~/.local/bin/uv").expanduser())
    if not Path(uv).exists():
        printer("uv bulunamadı: https://docs.astral.sh/uv/ ile kur")
        return 2
    target = venv_dir(worker)
    target.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target.parent).free / 1e9
    if free_gb < MIN_FREE_GB:
        printer(f"diskte {free_gb:.0f} GB boş; kurulum için en az {MIN_FREE_GB} GB gerekir")
        return 2
    env = dict(os.environ)
    env.pop("XDG_CACHE_HOME", None)  # uv's own cache stays in ~/.cache/uv
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
