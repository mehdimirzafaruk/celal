"""Global hotkey through Carbon ``RegisterEventHotKey`` (ctypes, no extra package).

Carbon hotkeys need no Accessibility permission. The handler runs on the main
run loop of the process that registered it (here: the menu bar process).
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections.abc import Callable

KEY_CODES = {"space": 49, "m": 46, "j": 38, "k": 40, "l": 37}
MODIFIERS = {"cmd": 0x100, "shift": 0x200, "option": 0x800, "alt": 0x800, "ctrl": 0x1000, "control": 0x1000}

_kEventClassKeyboard = int.from_bytes(b"keyb", "big")
_kEventHotKeyPressed = 5


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'ctrl+space' -> (key code, Carbon modifier mask)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts or parts[-1] not in KEY_CODES:
        raise ValueError(f"desteklenmeyen kısayol: {spec!r}")
    mods = 0
    for p in parts[:-1]:
        if p not in MODIFIERS:
            raise ValueError(f"desteklenmeyen değiştirici tuş: {p!r}")
        mods |= MODIFIERS[p]
    return KEY_CODES[parts[-1]], mods


class GlobalHotkey:
    def __init__(self, spec: str, callback: Callable[[], None]):
        self.spec = spec
        self._callback = callback
        self._handler = _HANDLER(self._on_event)  # keep a reference: Carbon holds a raw pointer
        self._ref = ctypes.c_void_p()
        self._handler_ref = ctypes.c_void_p()

    def _on_event(self, _call_ref, _event, _user) -> int:
        try:
            self._callback()
        except Exception:  # noqa: BLE001
            pass
        return 0

    def register(self) -> int:
        """Returns the Carbon OSStatus (0 = ok)."""
        code, mods = parse_hotkey(self.spec)
        carbon = ctypes.cdll.LoadLibrary(ctypes.util.find_library("Carbon"))
        carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
        target = carbon.GetApplicationEventTarget()
        spec = _EventTypeSpec(_kEventClassKeyboard, _kEventHotKeyPressed)
        carbon.InstallEventHandler.argtypes = [
            ctypes.c_void_p, _HANDLER, ctypes.c_ulong, ctypes.POINTER(_EventTypeSpec), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        ]
        status = carbon.InstallEventHandler(target, self._handler, 1, ctypes.byref(spec), None, ctypes.byref(self._handler_ref))
        if status != 0:
            return status
        carbon.RegisterEventHotKey.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, _EventHotKeyID, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)
        ]
        hk_id = _EventHotKeyID(int.from_bytes(b"CELL", "big"), 1)
        return carbon.RegisterEventHotKey(code, mods, hk_id, target, 0, ctypes.byref(self._ref))
