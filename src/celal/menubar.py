"""Menu bar icon, run as a child process (``python -m celal.menubar``).

pywebview owns the Cocoa run loop of the main process, and so does rumps, so
the icon lives in its own small process. Protocol over pipes, one line each:

- child → parent (stdout): ``toggle`` (show/hide window), ``mic``, ``stop``, ``quit``,
  ``hotkey <spec>`` / ``hotkey-failed <spec> <status>`` once at start
- parent → child (stdin): ``title <text>``; EOF means the parent is gone.
"""

from __future__ import annotations

import os
import sys
import threading


def _send(cmd: str) -> None:
    try:
        sys.stdout.write(cmd + "\n")
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        pass


def main() -> int:
    import rumps

    try:
        from AppKit import NSApplication

        NSApplication.sharedApplication().setActivationPolicy_(1)  # accessory: no Dock icon
    except Exception:  # noqa: BLE001
        pass

    state = {"title": "○ Celal", "quit": False}

    class CelalMenu(rumps.App):
        def __init__(self):
            super().__init__("Celal", title=state["title"], quit_button=None)
            self.menu = [
                rumps.MenuItem("Pencereyi göster / gizle", callback=lambda _: _send("toggle")),
                rumps.MenuItem(f"Mikrofon aç / kapat ({hotkey_label})", callback=lambda _: _send("mic")),
                rumps.MenuItem("Durdur (konuşmayı ve isteği kes)", callback=lambda _: _send("stop")),
                None,
                rumps.MenuItem("Çıkış", callback=self.quit),
            ]

        def quit(self, _):
            _send("quit")
            rumps.quit_application()

        @rumps.timer(0.5)
        def sync(self, _):
            if state["quit"]:
                rumps.quit_application()
            elif self.title != state["title"]:
                self.title = state["title"]

    def read_parent():
        for line in sys.stdin:
            line = line.strip()
            if line.startswith("title "):
                state["title"] = line[6:]
        state["quit"] = True  # parent closed the pipe

    threading.Thread(target=read_parent, daemon=True).start()

    hotkey_label = "kısayol yok"
    spec = os.environ.get("CELAL_HOTKEY", "ctrl+space")
    if spec and spec.lower() != "off":
        from celal.hotkey import GlobalHotkey

        hk = GlobalHotkey(spec, lambda: _send("mic"))
        try:
            status = hk.register()
        except Exception as e:  # noqa: BLE001
            status = str(e)
        if status == 0:
            hotkey_label = spec
            _send(f"hotkey {spec}")
        else:
            _send(f"hotkey-failed {spec} {status}")
    CelalMenu().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
