"""Local Mac API (Hermes → Mac): bearer auth, task start/status, busy, token file."""

import stat
import sys

import httpx
import pytest

from celal.mac_api import MacApiServer, ensure_token
from celal.mac_task import BUSY, MacTaskRunner
from fakes_mac import FakeClaude, Spawner, init_line, result_line, tool_line

TOKEN = "test-token-123"


@pytest.fixture
async def api(tmp_path):
    lines = [init_line(), tool_line("Bash", command="uname -a"), result_line("Darwin mac 27.0. Bitti.")]
    hold = {"on": False}
    spawner = Spawner(lambda: FakeClaude(lines if not hold["on"] else [init_line()], delay=0.02, hold=hold["on"]))
    runner = MacTaskRunner(claude=sys.executable, default_cwd=str(tmp_path), spawn=spawner)
    seen = []
    server = MacApiServer(runner, TOKEN, port=0, on_request=lambda k, d: seen.append(d))
    await server.start()
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{server.port}", timeout=5)
    yield server, client, runner, spawner, hold, seen
    await client.aclose()
    await runner.shutdown()
    await server.stop()


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


async def test_requires_bearer_token(api):
    server, client, runner, spawner, *_ = api
    for headers in ({}, auth("yanlis"), {"Authorization": TOKEN}):
        r = await client.post("/task", json={"task": "uname -a"}, headers=headers)
        assert r.status_code == 401
        r = await client.get("/health", headers=headers)
        assert r.status_code == 401
    assert spawner.calls == []  # nothing ran without the token


async def test_task_runs_and_reports(api, tmp_path):
    server, client, runner, spawner, hold, seen = api
    r = await client.get("/health", headers=auth())
    assert r.json() == {"ok": True, "busy": False, "running": 0, "max_parallel": 3}
    r = await client.post("/task", json={"task": "uname -a çalıştır"}, headers=auth())
    assert r.status_code == 202
    task_id = r.json()["task_id"]
    assert seen and seen[0]["task_id"] == task_id
    t = runner.get(task_id)
    assert t.source == "hermes"
    await t.wait(5)
    r = await client.get(f"/task/{task_id}", headers=auth())
    body = r.json()
    assert r.status_code == 200 and body["status"] == "done"
    assert body["result"] == "Darwin mac 27.0. Bitti."
    assert "komut çalıştırıyor: uname -a" in body["progress"]
    assert spawner.calls[0][1] == str(tmp_path)


async def test_busy_bad_input_and_unknown(api, tmp_path):
    server, client, runner, spawner, hold, seen = api
    hold["on"] = True
    r = await client.post("/task", json={"task": "uzun iş"}, headers=auth())
    assert r.status_code == 202
    r = await client.post("/task", json={"task": "ikinci"}, headers=auth())  # writer in the same (default) cwd
    assert r.status_code == 409 and "aynı klasörde" in r.json()["error"]
    await runner.stop()
    assert (await client.post("/task", content=b"{bozuk", headers=auth())).status_code == 400
    assert (await client.post("/task", json={"task": "  "}, headers=auth())).status_code == 400
    r = await client.post("/task", json={"task": "x", "cwd": str(tmp_path / "yok")}, headers=auth())
    assert r.status_code == 400 and "çalışma klasörü yok" in r.json()["error"]
    assert (await client.get("/task/yok", headers=auth())).status_code == 404
    assert (await client.get("/task", headers=auth())).status_code == 405


def test_token_file_created_private_and_stable(tmp_path):
    path = tmp_path / "celal" / "mac-api-key"
    t1 = ensure_token(path)
    assert len(t1) >= 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert ensure_token(path) == t1
    path.chmod(0o644)
    ensure_token(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_two_tasks_run_concurrently_with_separate_state(api, tmp_path):
    """Two POST /task calls while the first is still running → both 202, both running, independent."""
    server, client, runner, spawner, hold, seen = api
    hold["on"] = True
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    r1 = await client.post("/task", json={"task": "birinci", "cwd": str(tmp_path / "a")}, headers=auth())
    assert r1.status_code == 202, r1.text
    r2 = await client.post("/task", json={"task": "ikinci", "cwd": str(tmp_path / "b")}, headers=auth())
    assert r2.status_code == 202, r2.text  # was 409 with the single-task lock
    id1, id2 = r1.json()["task_id"], r2.json()["task_id"]
    assert id1 != id2
    h = (await client.get("/health", headers=auth())).json()
    assert h["busy"] is True and h["running"] == 2
    b1 = (await client.get(f"/task/{id1}", headers=auth())).json()
    b2 = (await client.get(f"/task/{id2}", headers=auth())).json()
    assert b1["status"] == "running" and b2["status"] == "running"
    assert b1["task"] == "birinci" and b2["task"] == "ikinci"
    assert b1["cwd"] != b2["cwd"]
    # stopping one does not touch the other
    assert await runner.stop(id1)
    b1 = (await client.get(f"/task/{id1}", headers=auth())).json()
    b2 = (await client.get(f"/task/{id2}", headers=auth())).json()
    assert b1["status"] == "stopped" and b2["status"] == "running"
    assert len(seen) == 2 and {d["task_id"] for d in seen} == {id1, id2}


async def test_read_only_tasks_share_a_cwd_but_writers_do_not(api, tmp_path):
    server, client, runner, spawner, hold, seen = api
    hold["on"] = True
    cwd = str(tmp_path)
    r1 = await client.post("/task", json={"task": "ekran görüntüsü al", "cwd": cwd, "access": "read"}, headers=auth())
    r2 = await client.post("/task", json={"task": "dosyaları listele", "cwd": cwd, "access": "read"}, headers=auth())
    assert (r1.status_code, r2.status_code) == (202, 202)
    assert r1.json()["access"] == "read"
    # a writer (default access) in the same folder is fine next to readers
    r3 = await client.post("/task", json={"task": "README düzelt", "cwd": cwd}, headers=auth())
    assert r3.status_code == 202 and r3.json()["access"] == "write"
    await runner.stop(r1.json()["task_id"])  # free a slot (limit is 3 by default)
    # a second writer in the same folder is refused, naming the running writer
    r4 = await client.post("/task", json={"task": "core.py düzelt", "cwd": cwd}, headers=auth())
    assert r4.status_code == 409
    assert r4.json()["conflict_task_id"] == r3.json()["task_id"]
    assert "aynı klasörde" in r4.json()["error"]
    # same folder spelled differently ("~"-style / trailing slash) is still the same folder
    r5 = await client.post("/task", json={"task": "x", "cwd": cwd + "/"}, headers=auth())
    assert r5.status_code == 409
    # another folder is fine
    (tmp_path / "other").mkdir()
    r6 = await client.post("/task", json={"task": "x", "cwd": str(tmp_path / "other")}, headers=auth())
    assert r6.status_code == 202
    assert (await client.post("/task", json={"task": "x", "access": "rw"}, headers=auth())).status_code == 400
    # once the writer is done, the folder is free again
    await runner.stop(r3.json()["task_id"])
    r7 = await client.post("/task", json={"task": "core.py düzelt", "cwd": cwd}, headers=auth())
    assert r7.status_code == 202


async def test_parallel_limit_returns_409_with_running_ids(api, tmp_path):
    server, client, runner, spawner, hold, seen = api
    hold["on"] = True
    runner.max_parallel = 2
    ids = []
    for i in range(2):
        d = tmp_path / f"d{i}"
        d.mkdir()
        r = await client.post("/task", json={"task": f"iş {i}", "cwd": str(d)}, headers=auth())
        assert r.status_code == 202
        ids.append(r.json()["task_id"])
    r = await client.post("/task", json={"task": "fazla"}, headers=auth())
    assert r.status_code == 409 and r.json()["error"] == BUSY
    assert r.json()["running_task_id"] in ids  # old field stays for Hermes
    assert sorted(r.json()["running_task_ids"]) == sorted(ids)
    assert (await client.get("/health", headers=auth())).json()["max_parallel"] == 2
