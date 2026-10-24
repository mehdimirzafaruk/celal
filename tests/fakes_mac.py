"""Fake brains for tab / core / API tests (no Claude process, no Hermes)."""

import asyncio

from celal.brains import Brain, TurnOutcome


class FakeBrains:
    """Brain factory: records prompts; "bekle" in a message blocks until ``release`` (or Durdur)."""

    def __init__(self, delay=0.0, replies=None):
        self.delay = delay
        self.replies = replies or {}
        self.created: list["FakeBrain"] = []
        self.prompts: list[tuple[str, str | None]] = []
        self.release = asyncio.Event()
        self.active = 0
        self.max_seen = 0
        self.warmed = 0

    def __call__(self, tab):
        b = FakeBrain(self, tab)
        self.created.append(b)
        return b

    def reply_for(self, text):
        for key, parts in self.replies.items():
            if key in (text or ""):
                return list(parts)
        return ["Tamam. ", f"{text} bitti."]


class FakeBrain(Brain):
    def __init__(self, owner: FakeBrains, tab):
        self.owner = owner
        self.kind = tab.kind
        self.stoppable = tab.kind == "mac"
        self.cwd = tab.cwd
        self.session_id = tab.session_id or f"sess-{tab.id}"
        self.resumed = tab.session_id
        self._connected = False
        self._stop = asyncio.Event()
        self.interrupts = 0
        self.closed = 0

    @property
    def connected(self):
        return self._connected

    async def warm(self):
        self._connected = True
        self.owner.warmed += 1

    async def turn(self, text, emit):
        self._connected = True
        self._stop.clear()
        self.owner.prompts.append((self.kind, text))
        self.owner.active += 1
        self.owner.max_seen = max(self.owner.max_seen, self.owner.active)
        try:
            emit("session", self.session_id)
            emit("progress", f"komut çalıştırıyor: {str(text)[:30]}")
            if text and "bekle" in text:
                waits = [asyncio.ensure_future(self.owner.release.wait()), asyncio.ensure_future(self._stop.wait())]
                await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
                for w in waits:
                    w.cancel()
            for part in self.owner.reply_for(text):
                if self._stop.is_set() and self.stoppable:
                    return TurnOutcome(False, interrupted=True, session_id=self.session_id)
                await asyncio.sleep(self.owner.delay)
                emit("text", part)
            if self._stop.is_set() and self.stoppable:
                return TurnOutcome(False, interrupted=True, session_id=self.session_id)
            return TurnOutcome(True, "".join(self.owner.reply_for(text)), session_id=self.session_id)
        finally:
            self.owner.active -= 1

    async def interrupt(self):
        self.interrupts += 1
        self._stop.set()

    async def close(self):
        self._connected = False
        self.closed += 1


async def until(cond, timeout=5.0, step=0.01):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if cond():
            return True
        await asyncio.sleep(step)
    return cond()
