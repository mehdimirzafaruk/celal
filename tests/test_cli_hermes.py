"""`celal hermes "<istek>"`: one Hermes request, answer on stdout; temporary tunnel only when needed."""

import io
import sys

import pytest

from celal import cli
from celal.hermes import Event, HermesError


class FakeHermesClient:
    instances = []
    online = True
    events = [Event("text", "Cron: "), Event("text", "3 iş, 2'si açık.")]
    error = None

    def __init__(self, settings, *, instructions=None, read_timeout=None, **kw):
        self.settings = settings
        self.instructions = instructions
        self.prompts = []
        self.closed = False
        FakeHermesClient.instances.append(self)

    async def ping(self):
        return (True, "bağlı") if FakeHermesClient.online else (False, "sunucuya bağlanılamadı")

    async def stream(self, text):
        self.prompts.append(text)
        for e in self.events:
            yield e
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


class FakeTunnel:
    started = 0
    stopped = 0

    def __init__(self, settings):
        self.status, self.detail = "açık", ""

    def start(self):
        FakeTunnel.started += 1
        FakeHermesClient.online = True  # the tunnel brings the server up

    async def stop(self):
        FakeTunnel.stopped += 1


@pytest.fixture
def fake(monkeypatch, tmp_path):
    import celal.hermes
    import celal.tunnel

    FakeHermesClient.instances = []
    FakeHermesClient.online = True
    FakeHermesClient.error = None
    FakeTunnel.started = FakeTunnel.stopped = 0
    monkeypatch.setattr(celal.hermes, "HermesClient", FakeHermesClient)
    monkeypatch.setattr(celal.tunnel, "TunnelManager", FakeTunnel)
    monkeypatch.setenv("CELAL_CONFIG", str(tmp_path / "yok.toml"))
    monkeypatch.setenv("CELAL_SSH_HOST", "user@example.invalid")
    monkeypatch.setenv("CELAL_MODELS_DIR", str(tmp_path / "yok"))  # `celal hermes` never needs the model disk
    return FakeHermesClient


def test_prints_the_answer(fake, capsys):
    assert cli.main(["hermes", "sunucudaki", "cron", "listesini", "ver"]) == 0
    out = capsys.readouterr()
    assert out.out == "Cron: 3 iş, 2'si açık.\n"
    client = fake.instances[0]
    assert client.prompts == ["sunucudaki cron listesini ver"]
    assert "Mac'e" in client.instructions and client.closed  # Hermes must not send it back to the Mac
    assert client.settings.conversation == "celal-mac-claude"
    assert FakeTunnel.started == 0  # Celal's tunnel was already up


def test_opens_a_temporary_tunnel_when_needed(fake, capsys):
    fake.online = False
    assert cli.main(["hermes", "uptime"]) == 0
    assert FakeTunnel.started == 1 and FakeTunnel.stopped == 1
    assert "3 iş" in capsys.readouterr().out


def test_errors_go_to_stderr_with_exit_code(fake, capsys, monkeypatch):
    fake.error = HermesError("sunucu hatası (HTTP 500)")
    fake.events = []
    assert cli.main(["hermes", "x"]) == 1
    assert "sunucu hatası" in capsys.readouterr().err
    fake.events = [Event("text", "Cron: "), Event("text", "3 iş, 2'si açık.")]
    fake.error = None
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["hermes"]) == 2  # empty request


def test_reads_the_request_from_stdin(fake, capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin'den istek\n"))
    assert cli.main(["hermes"]) == 0
    assert fake.instances[0].prompts == ["stdin'den istek"]
