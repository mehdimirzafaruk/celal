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
    assert r.json() == {"ok": True, "busy": False}
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
    r = await client.post("/task", json={"task": "ikinci"}, headers=auth())
    assert r.status_code == 409 and r.json()["error"] == BUSY
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
