"""Local Mac API (Hermes → Mac): bearer auth, non-blocking POST /task, queue, status, cancel, /tasks, 24 h results."""

import asyncio
import stat

import httpx
import pytest

from celal.mac_api import MacApiServer, ensure_token
from celal.tabs import TabManager
from fakes_mac import FakeBrains, until

TOKEN = "test-token-123"


@pytest.fixture
async def api(tmp_path):
    brains = FakeBrains()
    tabs = TabManager(brains, state_dir=tmp_path / "state", default_cwd=str(tmp_path), max_parallel=3)
    tabs.create("mac")  # the window's own tab
    seen = []
    server = MacApiServer(tabs, TOKEN, port=0, on_request=lambda k, d: seen.append(d))
    await server.start()
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{server.port}", timeout=5)
    yield server, client, tabs, brains, seen
    await client.aclose()
    brains.release.set()
    await tabs.shutdown()
    await server.stop()


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


async def test_requires_bearer_token(api):
    server, client, tabs, brains, _ = api
    for headers in ({}, auth("yanlis"), {"Authorization": TOKEN}):
        assert (await client.post("/task", json={"task": "uname -a"}, headers=headers)).status_code == 401
        assert (await client.get("/tasks", headers=headers)).status_code == 401
    assert brains.prompts == []  # nothing ran without the token


async def test_post_returns_at_once_and_runs_in_a_new_phone_tab(api, tmp_path):
    server, client, tabs, brains, seen = api
    active_before = tabs.active_id
    t0 = asyncio.get_running_loop().time()
    r = await client.post("/task", json={"task": "uname -a çalıştır bekle", "source": "telegram"}, headers=auth())
    assert asyncio.get_running_loop().time() - t0 < 1.0  # does not wait for the task
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "running" and body["queued"] is False and body["cwd"] == str(tmp_path)
    task_id, tab_id = body["task_id"], body["tab_id"]
    assert seen and seen[0]["task_id"] == task_id
    tab = tabs.get(tab_id)
    assert tab.source == "hermes" and tab.label == "uname -a çalıştır bekle"
    assert tabs.active_id == active_before  # 📱 tabs do not steal the window's focus
    assert await until(lambda: tab.status == "running")
    r = await client.get(f"/task/{task_id}", headers=auth())
    assert r.json()["status"] == "running" and r.json()["started_at"]
    brains.release.set()
    assert await until(lambda: tab.status == "done")
    body = (await client.get(f"/task/{task_id}", headers=auth())).json()
    assert body["status"] == "done" and body["result"] == "Tamam. uname -a çalıştır bekle bitti."
    assert body["finished_at"] >= body["started_at"]
    assert "komut çalıştırıyor: uname -a çalıştır bekle" in body["progress"]


async def test_two_parallel_posts_and_queue_beyond_the_limit(api):
    server, client, tabs, brains, _ = api
    rs = await asyncio.gather(*[client.post("/task", json={"task": f"iş {i} bekle"}, headers=auth())
                                for i in range(4)])
    bodies = [r.json() for r in rs]
    assert all(r.status_code == 202 for r in rs)
    assert sorted(b["queued"] for b in bodies) == [False, False, False, True]
    assert await until(lambda: len(tabs.running) == 3 and len(tabs.queued) == 1)
    listing = (await client.get("/tasks", headers=auth())).json()
    assert listing["running"] == 3 and listing["queued"] == 1 and listing["max_parallel"] == 3
    assert {t["task_id"] for t in listing["tasks"]} == {b["task_id"] for b in bodies}
    assert all(t["source"] == "hermes" for t in listing["tasks"])
    brains.release.set()
    for b in bodies:
        assert await until(lambda b=b: tabs.tasks.get(b["task_id"])["status"] == "done")
    assert brains.max_seen == 3


async def test_cancel_stops_the_task(api):
    server, client, tabs, brains, _ = api
    task_id = (await client.post("/task", json={"task": "uzun bekle"}, headers=auth())).json()["task_id"]
    assert await until(lambda: tabs.tasks.get(task_id)["status"] == "running")
    r = await client.post(f"/task/{task_id}/cancel", headers=auth())
    assert r.status_code == 200 and r.json()["cancelled"] is True
    assert await until(lambda: tabs.tasks.get(task_id)["status"] == "cancelled")
    assert (await client.get(f"/task/{task_id}", headers=auth())).json()["status"] == "cancelled"
    r = await client.post(f"/task/{task_id}/cancel", headers=auth())
    assert r.json()["cancelled"] is False


async def test_results_survive_a_restart(api, tmp_path):
    server, client, tabs, brains, _ = api
    task_id = (await client.post("/task", json={"task": "kısa iş"}, headers=auth())).json()["task_id"]
    assert await until(lambda: tabs.tasks.get(task_id)["status"] == "done")
    from celal.tabs import TaskStore

    again = TaskStore(tmp_path / "state" / "tasks.json")
    assert again.get(task_id)["result"] == "Tamam. kısa iş bitti."


async def test_bad_input_and_unknown(api, tmp_path):
    server, client, *_ = api
    assert (await client.post("/task", content=b"{bozuk", headers=auth())).status_code == 400
    assert (await client.post("/task", json={"task": "  "}, headers=auth())).status_code == 400
    r = await client.post("/task", json={"task": "x", "cwd": str(tmp_path / "yok")}, headers=auth())
    assert r.status_code == 400 and "çalışma klasörü yok" in r.json()["error"]
    assert (await client.get("/task/yok", headers=auth())).status_code == 404
    assert (await client.get("/task", headers=auth())).status_code == 405
    assert (await client.get("/health", headers=auth())).json()["ok"] is True


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
