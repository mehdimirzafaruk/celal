"""MacClaudeBrain against a fake ClaudeSDKClient that speaks real claude_agent_sdk message types."""

import asyncio

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TextBlock,
    ToolUseBlock,
)

from celal.brains import HermesBrain, MacClaudeBrain
from celal.hermes import Event, HermesError


def delta(text):
    return StreamEvent(uuid="u", session_id="s1", event={"type": "content_block_delta",
                                                         "delta": {"type": "text_delta", "text": text}})


def block_start():
    return StreamEvent(uuid="u", session_id="s1", event={"type": "content_block_start",
                                                         "content_block": {"type": "text", "text": ""}})


def result(text, aborted=False, error=False, sid="s1"):
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=error, num_turns=1,
                         session_id=sid, result=text, terminal_reason="aborted_streaming" if aborted else "completed")


INIT = SystemMessage(subtype="init", data={"session_id": "s1", "model": "claude-sonnet-test"})


class FakeSDK:
    """connect / query / receive_messages / interrupt / disconnect, scripted per query."""

    def __init__(self, options, scripts):
        self.options = options
        self.scripts = scripts  # list of lists of messages (or ("sleep", s) / ("wait", event))
        self.queries = []
        self.connected = False
        self.disconnected = False
        self.interrupted = 0
        self._q: asyncio.Queue = asyncio.Queue()

    async def connect(self):
        self.connected = True

    async def query(self, text):
        self.queries.append(text)
        script = self.scripts.pop(0) if self.scripts else []
        asyncio.get_running_loop().create_task(self._play(script))

    async def _play(self, script):
        for item in script:
            if isinstance(item, tuple) and item[0] == "sleep":
                await asyncio.sleep(item[1])
            elif isinstance(item, tuple) and item[0] == "wait":
                await item[1].wait()
            else:
                await self._q.put(item)

    def push_later(self, script):
        asyncio.get_running_loop().create_task(self._play(script))

    async def receive_messages(self):
        while True:
            yield await self._q.get()

    async def interrupt(self):
        self.interrupted += 1
        await self._q.put(result("", aborted=True))

    async def disconnect(self):
        self.disconnected = True


def brain_with(scripts, **kw):
    made = {}

    def factory(options):
        made["sdk"] = FakeSDK(options, scripts)
        return made["sdk"]

    b = MacClaudeBrain(cwd="/tmp", model="sonnet", client_factory=factory, options_factory=dict, **kw)
    return b, made


async def test_streams_text_progress_and_session():
    b, made = brain_with([[INIT, block_start(), delta("Bu klasörde "), delta("4 dosya var."),
                           AssistantMessage(content=[ToolUseBlock(id="t", name="Bash", input={"command": "ls"})],
                                            model="m"),
                           result("Bu klasörde 4 dosya var.")]])
    events = []
    out = await b.turn("kaç dosya var", lambda k, v: events.append((k, v)))
    assert out.ok and out.text == "Bu klasörde 4 dosya var." and out.session_id == "s1"
    assert [v for k, v in events if k == "text"] == ["Bu klasörde ", "4 dosya var."]
    assert ("progress", "komut çalıştırıyor: ls") in events
    assert ("session", "s1") in events
    assert made["sdk"].queries == ["kaç dosya var"]
    opts = made["sdk"].options
    assert opts["permission_mode"] == "bypassPermissions" and opts["include_partial_messages"] is True
    assert opts["model"] == "sonnet" and opts["cwd"] == "/tmp"
    assert "setting_sources" not in opts  # the user's ~/.claude settings, skills, hooks load
    assert opts["system_prompt"]["preset"] == "claude_code" and "hermes" in opts["system_prompt"]["append"]


async def test_full_text_used_when_no_partial_events_came():
    b, _ = brain_with([[AssistantMessage(content=[TextBlock(text="Tek parça cevap.")], model="m"), result("x")]])
    events = []
    await b.turn("soru", lambda k, v: events.append((k, v)))
    assert [v for k, v in events if k == "text"] == ["Tek parça cevap."]


async def test_follow_up_reuses_the_client_and_resume_after_close():
    b, made = brain_with([[result("bir")], [result("iki")]])
    await b.turn("1", lambda *a: None)
    sdk = made["sdk"]
    out = await b.turn("2", lambda *a: None)
    assert out.text == "iki" and made["sdk"] is sdk and sdk.queries == ["1", "2"]
    await b.close()
    assert sdk.disconnected and not b.connected
    assert b.options()["resume"] == "s1"  # the next connect resumes the same conversation


async def test_answer_waits_for_background_tasks():
    done = asyncio.Event()
    b, made = brain_with([[
        TaskStartedMessage(subtype="task_started", data={}, task_id="bg1", description="sleep 20",
                           uuid="u", session_id="s1"),
        result("Komut arka planda çalışıyor."),
        ("wait", done),
        TaskNotificationMessage(subtype="task_notification", data={}, task_id="bg1", status="completed",
                                output_file="/tmp/o", summary="ok", uuid="u", session_id="s1"),
        delta("X bitti."),
        result("X bitti."),
    ]])
    turn = asyncio.ensure_future(b.turn("sleep 20 && echo X", lambda *a: None))
    await asyncio.sleep(0.1)
    assert not turn.done()  # the first result is not the end: a background task is pending
    done.set()
    out = await asyncio.wait_for(turn, 2)
    assert out.ok and "X bitti." in out.text


async def test_interrupt_ends_the_turn():
    gate = asyncio.Event()
    b, made = brain_with([[delta("Başlıyorum."), ("wait", gate)]])
    turn = asyncio.ensure_future(b.turn("uzun iş", lambda *a: None))
    await asyncio.sleep(0.05)
    await b.interrupt()
    out = await asyncio.wait_for(turn, 2)
    assert out.interrupted and not out.ok and made["sdk"].interrupted == 1
    assert b.connected  # nothing left running: the client stays for the next message (no restart)


async def test_interrupt_with_background_task_closes_the_cli():
    gate = asyncio.Event()
    b, made = brain_with([[
        TaskStartedMessage(subtype="task_started", data={}, task_id="bg1", description="sleep", uuid="u",
                           session_id="s1"),
        result("arka planda"), ("wait", gate)]])
    turn = asyncio.ensure_future(b.turn("bekle", lambda *a: None))
    await asyncio.sleep(0.05)
    await b.interrupt()
    out = await asyncio.wait_for(turn, 5)
    assert out.interrupted and made["sdk"].disconnected and not b.connected


async def test_unprompted_answer_between_turns():
    b, made = brain_with([[result("tamam")]])
    await b.turn("başlat", lambda *a: None)
    made["sdk"].push_later([SystemMessage(subtype="hook_response", data={})])  # not an answer
    waiter = asyncio.ensure_future(b.wait_message())
    await asyncio.sleep(0.05)
    assert not waiter.done()
    made["sdk"].push_later([delta("Arka plan işi bitti."), result("Arka plan işi bitti.")])
    await asyncio.wait_for(waiter, 1)
    events = []
    out = await b.turn(None, lambda k, v: events.append((k, v)))
    assert out.text == "Arka plan işi bitti." and made["sdk"].queries == ["başlat"]


async def test_error_result():
    b, _ = brain_with([[result("API hatası", error=True)]])
    out = await b.turn("x", lambda *a: None)
    assert not out.ok and "API hatası" in out.error


class FakeHermes:
    def __init__(self, events=None, error=None):
        self.events, self.error, self.prompts = events or [], error, []

    async def stream(self, text):
        self.prompts.append(text)
        for e in self.events:
            yield e
        if self.error:
            raise self.error


async def test_hermes_brain_streams_and_reports_errors():
    h = FakeHermes([Event("tool", "terminal"), Event("text", "Disk "), Event("text", "yüzde 40 dolu.")])
    events = []
    out = await HermesBrain(h).turn("sunucuda disk", lambda k, v: events.append((k, v)))
    assert out.ok and out.text == "Disk yüzde 40 dolu." and ("progress", "Hermes araç kullanıyor: terminal") in events
    out = await HermesBrain(FakeHermes(error=HermesError("koptu"))).turn("x", lambda *a: None)
    assert not out.ok and out.error == "koptu"
    assert not HermesBrain(h).stoppable  # Durdur never cancels server work
