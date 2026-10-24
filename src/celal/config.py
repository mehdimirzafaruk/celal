"""Celal settings: ``~/.config/celal/config.toml`` overridden by ``CELAL_*`` env vars.

Nothing machine-specific is hardcoded: the SSH host and key come from the
config file (or env). The API key is read from a file and never logged.

Example ``~/.config/celal/config.toml``::

    [ssh]
    host = "ubuntu@sunucu.example"
    key = "~/.ssh/hermes_key"

    [hermes]
    local_port = 8642
    remote_port = 8642
    conversation = "celal-mac"
    api = "responses"          # veya "chat"

    [ui]
    hotkey = "ctrl+space"      # mikrofonu aç/kapat; "off" kapatır
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path("~/.config/celal")


@dataclass
class Settings:
    ssh_host: str | None = None
    ssh_key: str | None = None
    local_port: int = 8642
    remote_port: int = 8642
    conversation: str = "celal-mac"
    api: str = "responses"  # "responses" (server keeps history) | "chat"
    api_key_file: str = str(CONFIG_DIR / "api-key")
    model: str = "celal"
    hotkey: str = "ctrl+space"  # global mic toggle; "off" disables

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def read_api_key(self) -> str | None:
        path = Path(self.api_key_file).expanduser()
        try:
            key = path.read_text().strip()
        except OSError:
            return None
        return key or None


def load_settings(env: Mapping[str, str] | None = None, config_path: str | Path | None = None) -> Settings:
    env = os.environ if env is None else env
    path = Path(config_path or env.get("CELAL_CONFIG") or CONFIG_DIR / "config.toml").expanduser()
    data: dict = {}
    if path.is_file():
        with path.open("rb") as f:
            data = tomllib.load(f)
    ssh = data.get("ssh", {})
    hermes = data.get("hermes", {})
    s = Settings()
    s.ssh_host = env.get("CELAL_SSH_HOST") or ssh.get("host") or None
    s.ssh_key = env.get("CELAL_SSH_KEY") or ssh.get("key") or None
    s.local_port = int(env.get("CELAL_PORT") or hermes.get("local_port") or s.local_port)
    s.remote_port = int(env.get("CELAL_REMOTE_PORT") or hermes.get("remote_port") or s.remote_port)
    s.conversation = env.get("CELAL_CONVERSATION") or hermes.get("conversation") or s.conversation
    s.api = env.get("CELAL_API") or hermes.get("api") or s.api
    s.api_key_file = env.get("CELAL_API_KEY_FILE") or hermes.get("api_key_file") or s.api_key_file
    s.model = env.get("CELAL_MODEL") or hermes.get("model") or s.model
    s.hotkey = env.get("CELAL_HOTKEY") or data.get("ui", {}).get("hotkey") or s.hotkey
    if s.api not in ("responses", "chat"):
        raise ValueError(f"CELAL_API / hermes.api 'responses' ya da 'chat' olmalı, '{s.api}' geldi")
    return s
