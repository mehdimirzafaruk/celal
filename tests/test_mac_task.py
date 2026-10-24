"""Mac'te yap: request parsing, progress lines, summary, runner (fake and real subprocess)."""

import asyncio
import os
import stat
import sys
import time

import pytest

from celal.mac_task import (
    BUSY,
    CONFLICT,
    DONE,
    FAILED,
    READ,
    RUNNING,
    STOPPED,
    WRITE,
    MacBusy,
    MacConflict,
    MacTaskRunner,
    child_env,
    claude_command,
    describe_tool,
    match_prefix,
    parse_request,
    spoken_summary,
)
from fakes_mac import FakeClaude, Spawner, init_line, result_line, text_line, tool_line

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


def test_claude_command_flags():
    args = claude_command("/x/claude", "görev")
    assert args[:3] == ["/x/claude", "-p", "görev"]
    for flag in ("--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"):
        assert flag in args


def make_runner(tmp_path, make, events=None, **kw):
    spawner = Spawner(make)
    runner = MacTaskRunner(claude=sys.executable, default_cwd=str(tmp_path), spawn=spawner,
                           on_event=(lambda k, t: events.append((k, list(t.progress), t.status))) if events is not None else None,
                           **kw)
    return runner, spawner


async def test_stream_becomes_progress_and_result(tmp_path):
    events = []
    lines = [init_line(), tool_line("Bash", command="uv run pytest -q"), tool_line("Read", file_path="/r/README.md"),
             text_line("bitti"), result_line("40 test geçti. Hata yok. Ayrıntı aşağıda.")]
    runner, spawner = make_runner(tmp_path, lambda: FakeClaude(lines, delay=0.01), events)
    t = runner.start("testleri çalıştır")
    assert runner.busy
    assert await t.wait(5)
    assert t.status == DONE and t.result.startswith("40 test geçti")
    assert t.summary == "40 test geçti. Hata yok."
    assert t.progress[0].startswith("Claude başladı (claude-test")
    assert t.progress[1:] == ["komut çalıştırıyor: uv run pytest -q", "dosya okuyor: README.md"]
    kinds = [k for k, *_ in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("progress") == 3
    args, cwd, env = spawner.calls[0]
    assert cwd == str(tmp_path) and "--dangerously-skip-permissions" in args
    assert not runner.busy


async def test_slot_limit_frees_on_stop(tmp_path):
    runner, _ = make_runner(tmp_path, lambda: FakeClaude([init_line()], hold=True), max_parallel=1)
    t = runner.start("uzun iş")
    with pytest.raises(MacBusy, match=BUSY):
        runner.start("ikinci iş", access=READ)  # the limit counts readers too
    assert await runner.stop()
    assert t.status == STOPPED
    t2 = runner.start("şimdi olur")  # free again
    await runner.stop()
    assert t2.status == STOPPED


async def test_stop_kills_running_claude(tmp_path):
    procs = []

    def make():
        p = FakeClaude([init_line(), tool_line("Bash", command="sleep 100")], hold=True)
        procs.append(p)
        return p

    runner, _ = make_runner(tmp_path, make)
    t = runner.start("bekle")
    await asyncio.sleep(0.1)
    assert await runner.stop()
    assert procs[0].terminated or procs[0].killed
    assert t.status == STOPPED and t.summary == "Mac'teki işi durdurdum."


async def test_exit_without_result_is_a_failure(tmp_path):
    runner, _ = make_runner(tmp_path, lambda: FakeClaude([init_line()], rc=1, stderr=b"Error: not logged in\n"))
    t = runner.start("x")
    await t.wait(5)
    assert t.status == FAILED and "not logged in" in t.error and "kod 1" in t.error


async def test_error_result_and_timeout(tmp_path):
    runner, _ = make_runner(tmp_path, lambda: FakeClaude([result_line("API hatası", is_error=True)]))
    t = runner.start("x")
    await t.wait(5)
    assert t.status == FAILED and "API hatası" in t.error
    runner2, _ = make_runner(tmp_path, lambda: FakeClaude([init_line()], hold=True), timeout=0.2)
    t2 = runner2.start("y")
    await t2.wait(5)
    assert t2.status == FAILED and "zaman aşımı" in t2.error


async def test_missing_cwd_and_missing_claude(tmp_path):
    runner, _ = make_runner(tmp_path, lambda: FakeClaude([]))
    with pytest.raises(ValueError, match="çalışma klasörü yok"):
        runner.start("x", str(tmp_path / "yok"))
    runner.claude = str(tmp_path / "claude-yok")
    t = runner.start("x")
    await t.wait(5)
    assert t.status == FAILED and "claude komutu bulunamadı" in t.error


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_real_subprocess_group_is_killed(tmp_path):
    """A real child (shell script standing in for claude) and its own child die on stop()."""
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "fake-claude"
    script.write_text(
        "#!/bin/sh\n"
        'echo \'{"type":"system","subtype":"init","model":"m"}\'\n'
        f"sleep 300 & echo $! > {pidfile}\n"
        'echo \'{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"sleep 300"}}]}}\'\n'
        "wait\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    runner = MacTaskRunner(claude=str(script), default_cwd=str(tmp_path))
    t = runner.start("bekle")
    for _ in range(100):
        if len(t.progress) >= 2 and pidfile.exists() and pidfile.read_text().strip():
            break
        await asyncio.sleep(0.05)
    assert t.progress[-1] == "komut çalıştırıyor: sleep 300"
    child = int(pidfile.read_text())
    started = time.monotonic()
    assert await runner.stop()
    assert time.monotonic() - started < 5
    assert t.status == STOPPED
    await asyncio.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)  # the grandchild died with the group


async def test_real_subprocess_result(tmp_path):
    script = tmp_path / "fake-claude"
    script.write_text(
        "#!/bin/sh\n"
        'echo "cwd=$(pwd) args=$*" >&2\n'
        'echo \'{"type":"result","subtype":"success","is_error":false,"result":"Linux sunucu. İkinci cümle. Üçüncü.","num_turns":1}\'\n'
    )
    script.chmod(0o755)
    runner = MacTaskRunner(claude=str(script), default_cwd=str(tmp_path))
    t = runner.start("uname")
    assert await t.wait(10)
    assert t.status == DONE and t.summary == "Linux sunucu. İkinci cümle."


async def test_parallel_tasks_are_independent(tmp_path):
    """Up to max_parallel tasks run at once; finishing/stopping one leaves the others alone."""
    events = []
    procs = []

    def make():
        p = FakeClaude([init_line()], hold=True)
        procs.append(p)
        return p

    runner, _ = make_runner(tmp_path, make, events, max_parallel=3)
    dirs = []
    for i in range(3):
        d = tmp_path / f"p{i}"
        d.mkdir()
        dirs.append(str(d))
    t0 = runner.start("sıfır", dirs[0])
    t1 = runner.start("bir", dirs[1])
    t2 = runner.start("iki", dirs[2])
    assert len({t0.task_id, t1.task_id, t2.task_id}) == 3
    assert [t.task_id for t in runner.running] == [t0.task_id, t1.task_id, t2.task_id]
    with pytest.raises(MacBusy, match=BUSY):
        runner.start("üç", dirs[0])
    await asyncio.sleep(0.05)
    assert len(procs) == 3
    # finish t1 normally through its own process (result event)
    procs[1].stdout.feed(result_line("bir bitti."))
    assert await t1.wait(5)
    assert t1.status == DONE and t1.result == "bir bitti."
    assert t0.status == RUNNING and t2.status == RUNNING
    assert [t.task_id for t in runner.running] == [t0.task_id, t2.task_id]
    # stop only t2
    assert await runner.stop(t2.task_id)
    assert t2.status == STOPPED and t0.status == RUNNING
    assert procs[2].terminated and not procs[0].terminated
    # a slot is free again
    t3 = runner.start("üç", dirs[1])
    assert runner.busy and len(runner.running) == 2
    assert await runner.stop()  # no id: stop everything
    assert t0.status == STOPPED and t3.status == STOPPED and not runner.busy
    kinds = [k for k, *_ in events]
    assert kinds.count("start") == 4 and kinds.count("done") == 4


async def test_write_tasks_in_same_cwd_conflict(tmp_path):
    runner, _ = make_runner(tmp_path, lambda: FakeClaude([init_line()], hold=True), max_parallel=3)
    t = runner.start("README düzelt", str(tmp_path))
    assert t.access == WRITE
    with pytest.raises(MacConflict) as ei:
        runner.start("core.py düzelt", str(tmp_path))
    assert ei.value.task is t and CONFLICT in str(ei.value)
    assert isinstance(ei.value, MacBusy)
    # readers never conflict; a writer in another folder neither
    r = runner.start("ls", str(tmp_path), access=READ)
    assert r.access == READ
    (tmp_path / "x").mkdir()
    w = runner.start("x düzelt", str(tmp_path / "x"))
    assert len(runner.running) == 3
    with pytest.raises(ValueError, match="access"):
        runner.start("x", access="rw")
    await runner.stop(t.task_id)
    t2 = runner.start("core.py düzelt", str(tmp_path))  # writer gone → free
    assert t2.status == RUNNING
    await runner.stop()
