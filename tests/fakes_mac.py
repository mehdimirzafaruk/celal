"""Fake `claude -p --output-format stream-json` process for Mac task tests."""

import asyncio
import json


def init_line(model="claude-test"):
    return {"type": "system", "subtype": "init", "model": model, "cwd": "/tmp"}


def tool_line(name, **inp):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "input": inp}]}}


def text_line(text):
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def result_line(text, is_error=False):
    return {"type": "result", "subtype": "success", "is_error": is_error, "result": text, "num_turns": 3}


class FakeStream:
    def __init__(self, lines, delay=0.0, hold=False):
        self._lines = [(json.dumps(x, ensure_ascii=False) + "\n").encode() if isinstance(x, dict) else x for x in lines]
        self._delay = delay
        self._hold = hold  # keep the stream open after the last line (until closed)
        self.closed = asyncio.Event()

    async def readline(self):
        if self._lines:
            if self._delay:
                try:
                    await asyncio.wait_for(self.closed.wait(), self._delay)
                    return b""
                except TimeoutError:
                    pass
            return self._lines.pop(0)
        if self._hold:
            await self.closed.wait()
        return b""


class FakeErr:
    def __init__(self, data=b""):
        self.data = data

    async def read(self):
        return self.data


class FakeClaude:
    def __init__(self, lines, delay=0.0, hold=False, rc=0, stderr=b""):
        self.stdout = FakeStream(lines, delay, hold)
        self.stderr = FakeErr(stderr)
        self.returncode = None
        self.pid = None
        self._rc = rc
        self.killed = False
        self.terminated = False

    async def wait(self):
        while self.returncode is None:
            if self.stdout.closed.is_set() or (not self.stdout._lines and not self.stdout._hold):
                self.returncode = self._rc if not (self.killed or self.terminated) else -15
                break
            await asyncio.sleep(0.01)
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.stdout.closed.set()

    def kill(self):
        self.killed = True
        self.stdout.closed.set()


class Spawner:
    def __init__(self, make):
        self.make = make
        self.calls = []
        self.procs = []

    async def __call__(self, args, cwd, env):
        self.calls.append((list(args), cwd, env))
        p = self.make()
        self.procs.append(p)
        return p
