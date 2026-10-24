"""Helpers for the Mac brain: request parsing, Turkish progress lines, spoken summaries, child env.

The Mac's Claude Code itself runs through the Agent SDK (:mod:`celal.brains`);
this module keeps the small pure pieces around it.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

from celal.speech_text import clean_for_speech

def same_dir(a: str, b: str) -> bool:
    """True when two paths name the same directory (symlinks, ``..`` and trailing slashes resolved)."""
    return os.path.realpath(os.path.expanduser(a)) == os.path.realpath(os.path.expanduser(b))


# ---- parsing ---------------------------------------------------------------

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
_PATH_HINT = re.compile(
    r"^(?P<path>~?/\S+?)[,:]?\s+(?:içinde|içindeki|içinden|klasöründe|klasöründeki|dizininde|dizinindeki|"
    r"reposunda|reposundaki|projesinde|projesindeki)\b",
    re.IGNORECASE,
)


def match_prefix(text: str, prefixes: Sequence[str]) -> str | None:
    """Text after a "Mac'te"-style prefix, or None when the text has no prefix."""
    norm = text.strip().translate(_APOSTROPHES)
    low = norm.casefold()
    for p in sorted(prefixes, key=len, reverse=True):
        q = p.strip().translate(_APOSTROPHES).casefold()
        if not q or not low.startswith(q):
            continue
        rest = norm[len(q):]
        if rest and not (rest[0].isspace() or rest[0] in ",:;-–—"):
            continue  # "Macteki", "bilgisayardaki" are not the prefix
        rest = rest.lstrip(" \t\n,:;-–—")
        return rest
    return None


def split_cwd(task: str, home: Path | None = None) -> tuple[str, str | None]:
    """("~/Kod/x içinde testleri çalıştır") → (task, expanded dir) when the dir exists.

    The task text is kept whole (the path is useful context for Claude).
    """
    m = _PATH_HINT.match(task)
    if not m:
        return task, None
    raw = m.group("path").rstrip(",.:;")
    home = home or Path.home()
    path = Path(home, raw[2:]) if raw.startswith("~/") else Path(raw)
    if raw == "~":
        path = home
    return task, (str(path) if path.is_dir() else None)


def parse_request(text: str, prefixes: Sequence[str], *, home: Path | None = None) -> tuple[str, str | None] | None:
    """(task, cwd or None) when the text is a Mac request, else None."""
    rest = match_prefix(text, prefixes)
    if rest is None:
        return None
    return split_cwd(rest, home)


# ---- progress lines --------------------------------------------------------


def _short(value: str, n: int = 70) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= n else value[: n - 1] + "…"


def _base(path: str) -> str:
    return Path(str(path)).name or str(path)


def describe_tool(name: str, inp: dict | None) -> str:
    """One short Turkish line for a Claude Code tool call."""
    inp = inp or {}
    if name == "Bash":
        return f"komut çalıştırıyor: {_short(inp.get('command', ''))}"
    if name == "Read":
        return f"dosya okuyor: {_base(inp.get('file_path', ''))}"
    if name in ("Edit", "MultiEdit", "NotebookEdit"):
        return f"dosya düzenliyor: {_base(inp.get('file_path') or inp.get('notebook_path', ''))}"
    if name == "Write":
        return f"dosya yazıyor: {_base(inp.get('file_path', ''))}"
    if name == "Grep":
        return f"arıyor: {_short(inp.get('pattern', ''), 50)}"
    if name == "Glob":
        return f"dosya arıyor: {_short(inp.get('pattern', ''), 50)}"
    if name == "WebFetch":
        return f"web sayfası okuyor: {_short(inp.get('url', ''), 60)}"
    if name == "WebSearch":
        return f"web'de arıyor: {_short(inp.get('query', ''), 60)}"
    if name in ("Task", "Agent"):
        return f"alt ajan çalıştırıyor: {_short(inp.get('description', ''), 50)}"
    if name == "TodoWrite":
        return "plan güncelliyor"
    if name == "Skill":
        return f"skill kullanıyor: {_short(inp.get('skill') or inp.get('name', ''), 40)}"
    if name.startswith("mcp__"):
        return f"araç kullanıyor: {name.split('__')[-1]}"
    return f"araç kullanıyor: {name}"


def spoken_summary(text: str, sentences: int = 2, max_chars: int = 280) -> str:
    """First sentences of the result, cleaned for TTS (no model call)."""
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("```") or set(line) <= set("-=*_|: "):
            continue
        lines.append(line)
    flat = clean_for_speech(" ".join(lines))
    flat = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", flat)  # markdown links
    flat = " ".join(flat.split())
    if not flat:
        return ""
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9\"'(])", flat)
    out = " ".join(parts[:sentences]).strip()
    if len(out) > max_chars:
        cut = out[:max_chars].rsplit(" ", 1)[0]
        out = cut.rstrip(",;:") + "…"
    return out


# ---- child process ---------------------------------------------------------

_KEEP_CLAUDE_ENV = {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"}


def child_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Celal's environment minus Claude-session / API-billing variables.

    When Celal itself was started from inside a Claude Code session, its
    ``CLAUDECODE`` / ``CLAUDE_CODE_*`` / ``ANTHROPIC_BASE_URL`` variables would
    leak into the child; an ``ANTHROPIC_API_KEY`` would bill the API instead of
    the Max login. ``~/.local/bin`` is added to PATH (where the installer puts claude).
    """
    env = dict(os.environ if env is None else env)
    out = {
        k: v
        for k, v in env.items()
        if k in _KEEP_CLAUDE_ENV or not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC_"))
    }
    local_bin = str(Path(out.get("HOME") or Path.home()) / ".local" / "bin")
    path = out.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    if local_bin not in path.split(":"):
        out["PATH"] = f"{path}:{local_bin}"
    return out


def resolve_claude(name: str, env: dict[str, str] | None = None) -> str | None:
    expanded = os.path.expanduser(name)
    if os.sep in expanded:
        return expanded if os.access(expanded, os.X_OK) else None
    return shutil.which(expanded, path=(env or child_env()).get("PATH"))


# ---- routing ---------------------------------------------------------------

NEW = "new"  # "yeni görev …": a new tab
HERMES = "hermes"  # "sunucuda …" / "Hermes'e sor …": straight to Hermes
MAC = "mac"  # everything else: the active tab (Mac Claude by default)
MAC_FORCED = "mac!"  # old "Mac'te …" prefix: Mac even when the active tab talks to Hermes


def route(
    text: str,
    *,
    hermes_prefixes: Sequence[str],
    new_tab_prefixes: Sequence[str] = (),
    mac_prefixes: Sequence[str] = (),
) -> tuple[str, str]:
    """(target, text to send). Mac is the default; prefixes only force a new tab or Hermes."""
    text = text.strip()
    rest = match_prefix(text, new_tab_prefixes)
    if rest is not None and rest:
        return NEW, rest
    if match_prefix(text, hermes_prefixes) is not None:
        return HERMES, text
    rest = match_prefix(text, mac_prefixes)
    if rest is not None and rest:
        return MAC_FORCED, rest
    return MAC, text
