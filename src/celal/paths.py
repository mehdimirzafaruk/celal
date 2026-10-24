"""Model storage on the external disk.

Every model weight and cache (Whisper MLX, EMA Lightning, Hugging Face hub,
torch) lives under ``CELAL_MODELS_DIR`` (default ``/Volumes/KIOXIA/Modeller``).
If that folder is missing the disk is not plugged in: Celal stops with a clear
Turkish message instead of silently downloading gigabytes to the internal disk.

Call :func:`configure_model_storage` before importing anything that reads
``HF_HOME`` / ``XDG_CACHE_HOME`` at import time (huggingface_hub, ema_lightning,
mlx_whisper).
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

DEFAULT_MODELS_DIR = "/Volumes/KIOXIA/Modeller"


class ModelDiskMissing(RuntimeError):
    """The external model disk (``CELAL_MODELS_DIR``) is not available."""


def models_dir(env: MutableMapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("CELAL_MODELS_DIR") or DEFAULT_MODELS_DIR).expanduser()


def configure_model_storage(env: MutableMapping[str, str] | None = None) -> Path:
    """Point every model cache at the models dir; raise if the disk is missing."""
    env = os.environ if env is None else env
    root = models_dir(env)
    if not root.is_dir():
        raise ModelDiskMissing(
            f"model diski takılı değil: {root} bulunamadı. "
            "Harici diski tak ya da CELAL_MODELS_DIR ile başka bir klasör göster. "
            "Modeller iç diske indirilmez."
        )
    hf = root / "huggingface"
    env["HF_HOME"] = str(hf)
    env["HF_HUB_CACHE"] = str(hf / "hub")
    env["HF_XET_CACHE"] = str(hf / "xet")
    env["XDG_CACHE_HOME"] = str(root / "cache")  # ema_lightning batch-size cache
    env["TORCH_HOME"] = str(root / "torch")
    for sub in (hf / "hub", root / "cache", root / "torch"):
        sub.mkdir(parents=True, exist_ok=True)
    return root
