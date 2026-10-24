"""TunnelManager with a fake ssh process and fake port probe."""

import asyncio

from celal.tunnel import CONNECTING, DROPPED, EXTERNAL, NO_CONFIG, OPEN, STOPPED, TunnelManager, ssh_command


class FakeProc:
    def __init__(self):
        self.returncode = None
        self._done = asyncio.Event()
        self.killed = False
        self.terminated = False
        self.stderr = None

    async def wait(self):
        await self._done.wait()
        return self.returncode

    def exit(self, code=255):
        self.returncode = code
        self._done.set()

    def terminate(self):
        self.terminated = True
        self.exit(-15)

    def kill(self):
        self.killed = True
        self.exit(-9)


class World:
    """Port is open while one of our fake ssh processes is alive (or an external one listens)."""

    def __init__(self):
        self.procs: list[FakeProc] = []
        self.external = False
        self.args = []

    async def spawn(self, args):
        self.args.append(list(args))
        p = FakeProc()
        self.procs.append(p)
        return p

    async def port_open(self, port):
        return self.external or any(p.returncode is None for p in self.procs)


async def wait_for(cond, timeout=3.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        assert loop.time() < end, "condition not met"
        await asyncio.sleep(0.01)


def make(settings, world, statuses=None):
    return TunnelManager(
        settings,
        spawn=world.spawn,
        port_open=world.port_open,
        backoff=(0.01,),
        external_recheck=0.02,
        on_status=(lambda s, d: statuses.append(s)) if statuses is not None else None,
    )


def test_ssh_command_forwards_loopback_and_never_prompts(settings):
    args = ssh_command(settings)
    assert args[:4] == ["ssh", "-N", "-L", "18642:127.0.0.1:8642"]
    assert "BatchMode=yes" in args and "ExitOnForwardFailure=yes" in args
    assert args[-1] == "user@example.invalid"
    assert "-i" in args and not any("~" in a for a in args)  # key path expanded


async def test_restarts_ssh_when_it_drops(settings):
    world, statuses = World(), []
    tm = make(settings, world, statuses)
    tm.start()
    await wait_for(lambda: tm.status == OPEN)
    world.procs[0].exit(255)  # network drop
    await wait_for(lambda: len(world.procs) == 2 and tm.status == OPEN)
    assert DROPPED in statuses and statuses.count(CONNECTING) >= 2
    await tm.stop()
    assert world.procs[1].terminated and tm.status == STOPPED


async def test_kick_kills_stuck_ssh_and_restarts(settings):
    world = World()
    tm = make(settings, world)
    tm.start()
    await wait_for(lambda: tm.status == OPEN)
    assert tm.kick("test")
    assert world.procs[0].killed
    await wait_for(lambda: len(world.procs) == 2 and tm.status == OPEN)
    await tm.stop()


async def test_uses_existing_external_tunnel(settings):
    world = World()
    world.external = True
    tm = make(settings, world)
    tm.start()
    await wait_for(lambda: tm.status == EXTERNAL)
    await asyncio.sleep(0.05)
    assert world.procs == []
    world.external = False  # external tunnel went away → our own one starts
    await wait_for(lambda: tm.status == OPEN)
    assert len(world.procs) == 1
    await tm.stop()


async def test_no_host_configured(settings):
    settings.ssh_host = None
    world = World()
    tm = make(settings, world)
    tm.start()
    await wait_for(lambda: tm.status == NO_CONFIG)
    assert world.procs == []
    await tm.stop()


async def test_spawn_failure_is_retried(settings):
    world = World()
    calls = []

    async def bad_spawn(args):
        calls.append(1)
        if len(calls) < 3:
            raise FileNotFoundError("ssh yok")
        return await world.spawn(args)

    tm = TunnelManager(settings, spawn=bad_spawn, port_open=world.port_open, backoff=(0.01,))
    tm.start()
    await wait_for(lambda: tm.status == OPEN)
    assert len(calls) == 3
    await tm.stop()
