"""Where Celal's models live: on the Mac's own disk.

Every model weight and cache (Whisper MLX, Hugging Face hub, torch) lives
under ``CELAL_MODELS_DIR`` (default ``~/.local/share/celal/models``, created on
first use), so the mic works without any external disk. Voices have their own
folder (:mod:`celal.voices`). If ``CELAL_MODELS_DIR`` is set to a folder that
is missing (an external disk that is unplugged), Celal stops with a clear
Turkish message instead of downloading gigabytes somewhere else.

Call :func:`configure_model_storage` before importing anything that reads
``HF_HOME`` / ``XDG_CACHE_HOME`` at import time (huggingface_hub, mlx_whisper).
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

DEFAULT_MODELS_DIR = "~/.local/share/celal/models"


class ModelDiskMissing(RuntimeError):
    """``CELAL_MODELS_DIR`` points to a folder that is not there (unplugged disk)."""


def models_dir(env: MutableMapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("CELAL_MODELS_DIR") or DEFAULT_MODELS_DIR).expanduser()


def configure_model_storage(env: MutableMapping[str, str] | None = None) -> Path:
    """Point every model cache at the models dir; raise if a chosen folder is missing."""
    env = os.environ if env is None else env
    root = models_dir(env)
    if env.get("CELAL_MODELS_DIR"):
        if not root.is_dir():
            raise ModelDiskMissing(
                f"model diski takılı değil: {root} bulunamadı. "
                "Diski tak ya da CELAL_MODELS_DIR'ı kaldır (varsayılan Mac'in diski). "
                "Modeller başka yere indirilmez."
            )
    else:
        root.mkdir(parents=True, exist_ok=True)
    hf = root / "huggingface"
    env["HF_HOME"] = str(hf)
    env["HF_HUB_CACHE"] = str(hf / "hub")
    env["HF_XET_CACHE"] = str(hf / "xet")
    env["XDG_CACHE_HOME"] = str(root / "cache")
    env["TORCH_HOME"] = str(root / "torch")
    for sub in (hf / "hub", root / "cache", root / "torch"):
        sub.mkdir(parents=True, exist_ok=True)
    return root
