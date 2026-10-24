"""Mac brain helpers: routing rule, prefix parsing, progress lines, spoken summary, child env."""

import pytest

from celal.brains import scrub_parent_env
from celal.config import DEFAULT_HERMES_PREFIXES, DEFAULT_MAC_PREFIXES, DEFAULT_NEW_TAB_PREFIXES
from celal.mac_task import (
    HERMES,
    MAC,
    MAC_FORCED,
    NEW,
    child_env,
    describe_tool,
    match_prefix,
    parse_request,
    route,
    spoken_summary,
)
from celal.speech_text import SentenceBuffer

PREFIXES = ["Mac'te", "Mac'de", "Macte", "bilgisayarda"]


@pytest.mark.parametrize(
    "text,rest",
    [
        ("Mac'te testleri çalıştır", "testleri çalıştır"),
        ("mac'te testleri çalıştır", "testleri çalıştır"),
        ("MAC’TE: diski kontrol et", "diski kontrol et"),  # typographic apostrophe, colon
        ("Bilgisayarda, indirilenleri listele", "indirilenleri listele"),
        ("Macteki dosyalar ne durumda", None),  # not the prefix
        ("Hava nasıl?", None),
    ],
)
def test_prefix_matching(text, rest):
    assert match_prefix(text, PREFIXES) == rest


def test_path_override(tmp_path):
    (tmp_path / "Kod" / "celal").mkdir(parents=True)
    task, cwd = parse_request("Mac'te ~/Kod/celal içindeki testleri çalıştır", PREFIXES, home=tmp_path)
    assert task == "~/Kod/celal içindeki testleri çalıştır"  # path stays in the task as context
    assert cwd == str(tmp_path / "Kod" / "celal")
    # unknown dir → default cwd, task unchanged
    assert parse_request("Mac'te ~/yok/burada içinde bak", PREFIXES, home=tmp_path) == ("~/yok/burada içinde bak", None)
    # absolute path
    assert parse_request(f"bilgisayarda {tmp_path} klasöründe ls", PREFIXES)[1] == str(tmp_path)
    assert parse_request("merhaba", PREFIXES) is None


def test_progress_lines_are_short_turkish():
    assert describe_tool("Bash", {"command": "uv run pytest -q"}) == "komut çalıştırıyor: uv run pytest -q"
    assert describe_tool("Read", {"file_path": "/a/b/core.py"}) == "dosya okuyor: core.py"
    assert describe_tool("Edit", {"file_path": "/a/x.md"}) == "dosya düzenliyor: x.md"
    assert describe_tool("Grep", {"pattern": "TODO"}) == "arıyor: TODO"
    assert describe_tool("WebSearch", {"query": "hava"}) == "web'de arıyor: hava"
    assert describe_tool("mcp__srv__do_thing", {}) == "araç kullanıyor: do_thing"
    long = describe_tool("Bash", {"command": "x" * 500})
    assert len(long) < 100 and long.endswith("…")


def test_spoken_summary_takes_two_sentences_without_markdown():
    text = "**42 test geçti**, hiç hata yok. Süre 12 saniye. Ayrıntılar:\n\n- test_a\n- test_b\n```\nlog\n```"
    assert spoken_summary(text) == "42 test geçti, hiç hata yok. Süre 12 saniye."
    assert spoken_summary("") == ""
    assert len(spoken_summary("kelime " * 200)) <= 281


def test_child_env_drops_claude_session_and_api_billing_vars(tmp_path):
    env = child_env({
        "HOME": str(tmp_path), "PATH": "/usr/bin", "CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "x",
        "ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL": "http://x", "CLAUDE_CODE_OAUTH_TOKEN": "t", "LANG": "tr",
    })
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_ENTRYPOINT" not in env
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_BASE_URL" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "t" and env["LANG"] == "tr"
    assert env["PATH"].endswith(str(tmp_path / ".local" / "bin"))


def r(text):
    return route(text, hermes_prefixes=DEFAULT_HERMES_PREFIXES, new_tab_prefixes=DEFAULT_NEW_TAB_PREFIXES,
                 mac_prefixes=DEFAULT_MAC_PREFIXES)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("bu klasörde kaç dosya var", (MAC, "bu klasörde kaç dosya var")),  # Mac is the default
        ("sunucudaki hermes cron listesini söyle", (MAC, "sunucudaki hermes cron listesini söyle")),  # Claude decides
        ("sunucuda uptime ne", (HERMES, "sunucuda uptime ne")),
        ("Hermes'e sor: Yasomi videosu hazır mı", (HERMES, "Hermes'e sor: Yasomi videosu hazır mı")),
        ("Sunucuda, disk ne kadar dolu", (HERMES, "Sunucuda, disk ne kadar dolu")),
        ("yeni görev README'yi güncelle", (NEW, "README'yi güncelle")),
        ("Yeni görev: sunucuda log bak", (NEW, "sunucuda log bak")),
        ("Mac'te testleri çalıştır", (MAC_FORCED, "testleri çalıştır")),  # old prefix: Mac, prefix dropped
        ("yeni görevler listesi", (MAC, "yeni görevler listesi")),
    ],
)
def test_routing_rule_mac_first(text, expected):
    assert r(text) == expected


def test_scrub_parent_env_keeps_login_token():
    env = {"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "x", "ANTHROPIC_BASE_URL": "http://proxy",
           "ANTHROPIC_API_KEY": "k", "CLAUDE_CODE_OAUTH_TOKEN": "t", "CLAUDE_CONFIG_DIR": "/c", "PATH": "/bin"}
    removed = scrub_parent_env(env)
    assert set(removed) == {"CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"}
    assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "t", "CLAUDE_CONFIG_DIR": "/c", "PATH": "/bin"}


def test_sentence_buffer_speaks_whole_sentences_and_skips_code():
    b = SentenceBuffer()
    out = []
    for d in ["Bu klasörde ", "4 dosya var. Biri ", "**README.md**.\n\n```\nls -la\n```\n| a | b |\n", "Son cümle"]:
        out += b.feed(d)
    assert out == ["Bu klasörde 4 dosya var.", "Biri README.md."]
    assert b.flush() == ["Son cümle"]
