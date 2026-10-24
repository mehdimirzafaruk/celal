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

    [mac]                      # "Mac'te yap": görevi Mac'teki Claude Code'a ver
    prefixes = ["Mac'te", "bilgisayarda"]   # büyük/küçük harf fark etmez
    cwd = "~"                  # varsayılan çalışma klasörü
    claude = "claude"          # yol ya da PATH'teki ad (~/.local/bin de aranır)
    timeout = 1800             # saniye
    max_parallel = 3           # aynı anda en çok kaç Mac işi çalışır
    api = true                 # Hermes → Mac yerel API'si (127.0.0.1:api_port)
    api_port = 8643
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path("~/.config/celal")
DEFAULT_MAC_PREFIXES = ("Mac'te", "Mac'de", "Macte", "bilgisayarda")


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "hayır", "")


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
    # "Mac'te yap": tasks for the Mac's own Claude Code
    mac_prefixes: list[str] = field(default_factory=lambda: list(DEFAULT_MAC_PREFIXES))
    mac_cwd: str = "~"
    mac_claude: str = "claude"
    mac_timeout: float = 1800.0
    mac_max_parallel: int = 3  # concurrent "claude -p" tasks (never below 1)
    mac_api: bool = True  # local HTTP API for Hermes (reached through ssh -R)
    mac_api_port: int = 8643
    mac_api_key_file: str = str(CONFIG_DIR / "mac-api-key")

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
    mac = data.get("mac", {})
    prefixes = env.get("CELAL_MAC_PREFIXES")
    if prefixes:
        s.mac_prefixes = [p.strip() for p in prefixes.split(",") if p.strip()]
    elif isinstance(mac.get("prefixes"), list):
        s.mac_prefixes = [str(p).strip() for p in mac["prefixes"] if str(p).strip()]
    s.mac_cwd = env.get("CELAL_MAC_CWD") or mac.get("cwd") or s.mac_cwd
    s.mac_claude = env.get("CELAL_CLAUDE") or mac.get("claude") or s.mac_claude
    s.mac_timeout = float(env.get("CELAL_MAC_TIMEOUT") or mac.get("timeout") or s.mac_timeout)
    raw_parallel = env.get("CELAL_MAC_MAX_PARALLEL") or mac.get("max_parallel")
    if raw_parallel is not None and str(raw_parallel).strip() != "":
        s.mac_max_parallel = max(1, int(raw_parallel))
    if "CELAL_MAC_API" in env:
        s.mac_api = _bool(env["CELAL_MAC_API"])
    elif "api" in mac:
        s.mac_api = _bool(mac["api"])
    s.mac_api_port = int(env.get("CELAL_MAC_PORT") or mac.get("api_port") or s.mac_api_port)
    s.mac_api_key_file = env.get("CELAL_MAC_API_KEY_FILE") or mac.get("api_key_file") or s.mac_api_key_file
    if s.api not in ("responses", "chat"):
        raise ValueError(f"CELAL_API / hermes.api 'responses' ya da 'chat' olmalı, '{s.api}' geldi")
    return s
