"""TabManager: per-tab queue, shared parallel limit, Durdur, idle close, persistence; TaskStore 24 h."""

import asyncio
import json

from celal.tabs import DONE, ERROR, HERMES, MAC, QUEUED, RUNNING, STOPPED, TabManager, TaskStore, make_label
from fakes_mac import FakeBrains, until


def manager(tmp_path, brains=None, **kw):
    events = []
    m = TabManager(brains or FakeBrains(), state_dir=tmp_path, default_cwd=str(tmp_path),
                   on_event=lambda k, t, d: events.append((k, t.id, d)), **kw)
    return m, events


def test_label_is_first_words():
    assert make_label("README dosyasını güncelle ve testleri çalıştır") == "README dosyasını güncelle ve"
    assert make_label("   ") == "Yeni görev"
    assert len(make_label("çok" * 40)) <= 28


async def test_message_answered_in_its_tab_and_label_set(tmp_path):
    m, events = manager(tmp_path)
    tab = m.create(MAC)
    assert m.submit(tab, "bu klasörde kaç dosya var") == RUNNING
    assert await until(lambda: tab.status == DONE)
    assert tab.label == "bu klasörde kaç dosya"
    roles = [ln["role"] for ln in tab.lines]
    assert roles == ["user", "progress", "assistant"]
    assert tab.lines[-1]["text"] == "Tamam. bu klasörde kaç dosya var bitti."
    kinds = [k for k, *_ in events]
    assert "turn_start" in kinds and "delta" in kinds and "turn_end" in kinds
    assert tab.session_id == f"sess-{tab.id}"
    await m.shutdown()


async def test_follow_up_queues_behind_the_running_answer_same_brain(tmp_path):
    brains = FakeBrains()
    m, _ = manager(tmp_path, brains)
    tab = m.create(MAC)
    m.submit(tab, "uzun iş bekle")
    assert await until(lambda: tab.status == RUNNING)
    assert m.submit(tab, "takip sorusu") == QUEUED  # busy tab: "sırada"
    assert tab.waiting == 1 and tab.lines[-1]["queued"] is True
    brains.release.set()
    assert await until(lambda: tab.finished == 2)
    assert [p for _, p in brains.prompts] == ["uzun iş bekle", "takip sorusu"]
    assert len(brains.created) == 1  # the same client answered both: no new process
    await m.shutdown()


async def test_at_most_max_parallel_tabs_run_the_rest_wait(tmp_path):
    brains = FakeBrains()
    m, _ = manager(tmp_path, brains, max_parallel=3)
    tabs = [m.create(MAC, activate=False) for _ in range(4)]
    states = [m.submit(t, f"görev {i} bekle") for i, t in enumerate(tabs)]
    assert states == [RUNNING, RUNNING, RUNNING, QUEUED]
    assert await until(lambda: len(m.running) == 3)
    await asyncio.sleep(0.1)
    assert len(m.running) == 3 and len(m.queued) == 1 and brains.max_seen == 3
    assert tabs[3].status == QUEUED
    brains.release.set()
    assert await until(lambda: all(t.status == DONE for t in tabs))
    assert brains.max_seen == 3
    await m.shutdown()


async def test_durdur_interrupts_mac_tab_and_drops_its_queue(tmp_path):
    brains = FakeBrains()
    m, _ = manager(tmp_path, brains)
    tab = m.create(MAC)
    m.submit(tab, "uzun iş bekle")
    m.submit(tab, "sonraki")
    assert await until(lambda: tab.status == RUNNING)
    res = await m.stop(tab)
    assert res == {"stopped": True, "dropped": 1}
    assert tab.status == STOPPED and brains.created[0].interrupts == 1
    assert tab.lines[-1] == {**tab.lines[-1], "role": "info", "text": "durduruldu"}
    await asyncio.sleep(0.1)
    assert [p for _, p in brains.prompts] == ["uzun iş bekle"]  # the queued message never ran
    # the tab still works afterwards
    m.submit(tab, "yeni soru")
    assert await until(lambda: tab.status == DONE)
    await m.shutdown()


async def test_durdur_on_a_queued_tab_cancels_the_wait(tmp_path):
    brains = FakeBrains()
    m, _ = manager(tmp_path, brains, max_parallel=1)
    a, b = m.create(MAC), m.create(MAC)
    m.submit(a, "a bekle")
    assert m.submit(b, "b") == QUEUED
    assert await until(lambda: a.status == RUNNING and b.status == QUEUED)
    assert (await m.stop(b))["stopped"]
    assert b.status == STOPPED
    brains.release.set()
    assert await until(lambda: a.status == DONE)
    assert [p for _, p in brains.prompts] == ["a bekle"]
    await m.shutdown()


async def test_durdur_never_interrupts_hermes_work(tmp_path):
    brains = FakeBrains()
    m, _ = manager(tmp_path, brains)
    tab = m.create(HERMES)
    m.submit(tab, "sunucuda bekle")
    assert await until(lambda: tab.status == RUNNING)
    res = await m.stop(tab)
    assert res["stopped"] is False and res["server"] is True
    assert brains.created[0].interrupts == 0 and tab.status == RUNNING
    brains.release.set()
    assert await until(lambda: tab.status == DONE)  # the server's answer still arrives
    await m.shutdown()


async def test_idle_client_closes_and_keeps_session(tmp_path):
    brains = FakeBrains()
    m, _ = manager(tmp_path, brains, idle_close=0.1)
    tab = m.create(MAC)
    m.submit(tab, "merhaba")
    assert await until(lambda: tab.status == DONE)
    brain = brains.created[0]
    assert await until(lambda: brain.closed == 1, 2)
    assert tab.session_id == brain.session_id
    m.submit(tab, "yine ben")
    assert await until(lambda: tab.finished == 2)
    assert len(brains.created) == 1 and brain.connected  # reconnects (resume) on the next message
    await m.shutdown()


async def test_error_outcome_and_exception(tmp_path):
    class Boom(FakeBrains):
        def __call__(self, tab):
            b = super().__call__(tab)

            async def turn(text, emit):
                raise RuntimeError("çöktü")

            b.turn = turn
            return b

    m, _ = manager(tmp_path, Boom())
    tab = m.create(MAC)
    m.submit(tab, "x")
    assert await until(lambda: tab.status == ERROR)
    assert "çöktü" in tab.lines[-1]["text"]
    await m.shutdown()


async def test_tabs_survive_restart(tmp_path):
    m, _ = manager(tmp_path)
    a = m.create(MAC, cwd=str(tmp_path))
    m.submit(a, "ilk görev")
    assert await until(lambda: a.status == DONE)
    b = m.create(HERMES, label="Hermes: test")
    await m.shutdown()
    data = json.loads((tmp_path / "tabs.json").read_text())
    assert data["active"] == b.id and len(data["tabs"]) == 2

    brains = FakeBrains()
    m2, _ = manager(tmp_path, brains)
    assert m2.restore() == 2
    a2 = m2.get(a.id)
    assert a2.label == "ilk görev" and a2.session_id == a.session_id and a2.status == DONE
    assert [ln["role"] for ln in a2.lines] == ["user", "progress", "assistant"]
    assert m2.active_id == b.id
    m2.submit(a2, "devam")
    assert await until(lambda: a2.finished == 1)
    assert brains.created[0].resumed == a.session_id  # resumes the saved Claude session
    await m2.shutdown()


async def test_close_tab_stops_it_and_moves_active(tmp_path):
    brains = FakeBrains()
    m, _ = manager(tmp_path, brains)
    a, b = m.create(MAC), m.create(MAC)
    m.submit(b, "bekle")
    assert await until(lambda: b.status == RUNNING)
    assert await m.close(b.id)
    assert b.id not in m.tabs and m.active_id == a.id
    assert brains.created[0].interrupts == 1
    await m.shutdown()


def test_task_store_keeps_results_24h(tmp_path):
    now = [1000.0]
    path = tmp_path / "tasks.json"
    store = TaskStore(path, clock=lambda: now[0])

    class T:
        id, label, cwd, source = "t1", "etiket", "/tmp", "hermes"

    rec = store.create(T, "iş", queued=False)
    store.update(rec["task_id"], status="done", result="sonuç", finished_at=now[0])
    again = TaskStore(path, clock=lambda: now[0] + 3600)
    assert again.get(rec["task_id"])["result"] == "sonuç"
    later = TaskStore(path, clock=lambda: now[0] + 25 * 3600)
    assert later.get(rec["task_id"]) is None


def test_task_store_marks_interrupted_running_tasks(tmp_path):
    path = tmp_path / "tasks.json"
    store = TaskStore(path)

    class T:
        id, label, cwd, source = "t1", "etiket", "/tmp", "hermes"

    rec = store.create(T, "iş", queued=False)
    again = TaskStore(path)
    assert again.get(rec["task_id"])["status"] == "error"
