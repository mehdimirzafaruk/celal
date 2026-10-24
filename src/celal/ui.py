"""Native window (pywebview, one dark HTML page) + menu bar icon (rumps child process)."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
from importlib import resources

from loguru import logger

from celal.config import Settings
from celal.core import CelalCore
from celal.pipeline import Config


class Bridge:
    """Delivers core events to the page from one thread, in order, batched."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._ready = False
        self._backlog: list[dict] = []
        self.window = None
        self.on_status = None
        threading.Thread(target=self._pump, name="ui-bridge", daemon=True).start()

    def emit(self, ev: dict) -> None:
        if ev.get("type") == "status" and self.on_status:
            self.on_status(ev)
        self._q.put(ev)

    def mark_ready(self, snapshot: dict, history: dict | None = None) -> list[dict]:
        with self._lock:
            self._ready = True
            backlog, self._backlog = self._backlog, []
        # State (tab list), history (rebuilds the tab logs), what happened meanwhile, the state again.
        backlog = [e for e in backlog if e.get("type") not in ("tab_line", "tab_delta", "tab_break")]
        return [snapshot] + ([history] if history else []) + backlog + [snapshot]

    def _pump(self) -> None:
        while True:
            batch = [self._q.get()]
            try:
                while len(batch) < 100:
                    batch.append(self._q.get_nowait())
            except queue.Empty:
                pass
            with self._lock:
                if not self._ready or self.window is None:
                    self._backlog.extend(batch)
                    continue
            try:
                self.window.evaluate_js(f"window.celalEvents({json.dumps(batch, ensure_ascii=False)})")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"evaluate_js failed: {e}")


class Api:
    """Methods callable from the page as ``window.pywebview.api.*``.

    pywebview runs each call on its own thread; everything is handed to the
    core loop with :meth:`CelalCore.call` and returns at once.
    """

    def __init__(self, core: CelalCore, bridge: Bridge):
        self._core = core
        self._bridge = bridge

    def ready(self):
        history = {"type": "history", "tabs": self._core.history()}
        return self._bridge.mark_ready(self._core.snapshot(), history)

    def send_text(self, text: str):
        self._core.call(self._core.send_text(str(text)))
        return True

    def new_tab(self):
        self._core.call(self._core.new_tab())
        return True

    def select_tab(self, tab_id: str):
        self._core.call(self._core.select_tab(str(tab_id)))
        return True

    def close_tab(self, tab_id: str):
        self._core.call(self._core.close_tab(str(tab_id)))
        return True

    def set_mic(self, on: bool):
        self._core.call(self._core.set_mic(bool(on)))
        return True

    def interrupt(self):
        logger.info("Durdur (pencere)")
        self._core.call(self._core.interrupt())
        return True


class MenuBar:
    def __init__(self, on_command, hotkey: str = "ctrl+space"):
        self._on_command = on_command
        self._hotkey = hotkey
        self._proc: subprocess.Popen | None = None

    def start(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "celal.menubar"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "CELAL_HOTKEY": self._hotkey},
            )
        except OSError as e:
            logger.warning(f"menü çubuğu simgesi başlatılamadı: {e}")
            return False
        threading.Thread(target=self._read, name="menubar-read", daemon=True).start()
        return True

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            cmd = line.strip()
            if cmd:
                self._on_command(cmd)

    def set_title(self, title: str) -> None:
        if self.alive and self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(f"title {title}\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            self._proc.kill()


def menubar_title(ev: dict) -> str:
    """'● Celal ⏺' listening, '● Celal ◌' mic open but passive (voice commands), '○' not connected."""
    mark = "●" if ev.get("connected") else "○"
    if not ev.get("mic"):
        return f"{mark} Celal"
    wake = ev.get("wake") or {}
    passive = wake.get("on") and not wake.get("active")
    return f"{mark} Celal {'◌' if passive else '⏺'}"


def _set_app_name(name: str) -> None:
    try:
        from Foundation import NSBundle

        info = NSBundle.mainBundle().localizedInfoDictionary() or NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = name
    except Exception:  # noqa: BLE001
        pass


def run_window(
    settings: Settings,
    cfg: Config,
    *,
    debug: bool = False,
    menubar: bool = True,
    voice: bool = True,
    notice: str | None = None,
) -> int:
    import webview

    bridge = Bridge()

    def printer(line: str) -> None:
        # Console lines ([sen], [celal], ⏱) go to the log; timing also to the page.
        logger.info(line)
        if line.startswith("⏱"):
            bridge.emit({"type": "info", "text": line})

    cfg.printer = printer
    cfg.print_sentences = False
    core = CelalCore(settings, cfg, voice=voice, emit=bridge.emit)

    loop = asyncio.new_event_loop()
    started = threading.Event()
    start_error: list[BaseException] = []

    def loop_main():
        asyncio.set_event_loop(loop)

        async def boot():
            try:
                await core.start()
            except BaseException as e:  # noqa: BLE001
                start_error.append(e)
                logger.exception("Celal başlatılamadı")
                core.emit({"type": "error", "text": f"Celal başlatılamadı: {e}"})
            finally:
                started.set()
                if notice:
                    core.emit({"type": "error", "text": notice})

        loop.create_task(boot())
        loop.run_forever()

    threading.Thread(target=loop_main, name="celal-core", daemon=True).start()

    html = resources.files("celal").joinpath("web/index.html").read_text(encoding="utf-8")
    api = Api(core, bridge)
    window = webview.create_window(
        "Celal",
        html=html,
        js_api=api,
        width=480,
        height=760,
        min_size=(360, 480),
        background_color="#0f1115",
    )
    bridge.window = window
    ui = {"visible": True, "quitting": False}

    def toggle():
        if ui["visible"]:
            window.hide()
        else:
            window.show()
        ui["visible"] = not ui["visible"]

    def quit_app():
        ui["quitting"] = True
        window.destroy()

    def on_menu(cmd: str):
        if cmd == "toggle":
            toggle()
        elif cmd == "mic":
            core.call(core.set_mic(not core.mic))
        elif cmd == "stop":
            logger.info("Durdur (menü)")
            core.call(core.interrupt())
        elif cmd == "quit":
            quit_app()
        elif cmd.startswith("hotkey-failed "):
            _, spec, status = (cmd.split(" ", 2) + ["", ""])[:3]
            bridge.emit({"type": "error", "text": f"Kısayol {spec} kaydedilemedi ({status}); mikrofon düğmesini kullan."})
        elif cmd.startswith("hotkey "):
            spec = cmd.split(" ", 1)[1]
            bridge.emit({"type": "hotkey", "spec": spec})

    bar = MenuBar(on_menu, hotkey=settings.hotkey)
    if menubar:
        bar.start()

    bridge.on_status = lambda ev: bar.set_title(menubar_title(ev))

    def on_closing():
        # Closing the window hides it while the menu bar icon is there to bring it back.
        if ui["quitting"] or not bar.alive:
            return True
        ui["visible"] = False
        threading.Thread(target=window.hide, daemon=True).start()
        return False

    window.events.closing += on_closing
    _set_app_name("Celal")

    import signal

    def on_sigint(*_):
        # Ctrl-C in the terminal: same clean exit as the menu's "Çıkış".
        logger.info("sinyal alındı, çıkılıyor")
        threading.Thread(target=quit_app, daemon=True).start()

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    def keep_python_ticking():
        # Python signal handlers only run when the main thread executes Python;
        # inside the Cocoa run loop that needs a periodic no-op on the main thread.
        try:
            from PyObjCTools import AppHelper
        except ImportError:
            return

        def tick():
            ui["ticks"] = ui.get("ticks", 0) + 1
            if ui["ticks"] == 1:
                # Cocoa replaces the C-level handlers during startup; install ours again.
                signal.signal(signal.SIGINT, on_sigint)
                signal.signal(signal.SIGTERM, on_sigint)
            if not ui["quitting"]:
                AppHelper.callLater(0.3, tick)

        AppHelper.callAfter(tick)

    try:
        webview.start(keep_python_ticking, debug=debug)
    finally:
        bar.stop()
        try:
            asyncio.run_coroutine_threadsafe(core.shutdown(), loop).result(timeout=15)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"shutdown: {e}")
            core.tunnel.kill_now()
        loop.call_soon_threadsafe(loop.stop)
    return 1 if start_error else 0
