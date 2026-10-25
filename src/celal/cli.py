"""Command line: ``celal`` (window), ``celal --text`` (terminal), ``celal --bench``."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from loguru import logger


def _setup_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


async def _run_text(settings, cfg, args) -> int:
    from celal.core import CelalCore
    from celal.pipeline import OUT_RATE

    def show(ev: dict) -> None:
        kind = ev.get("type")
        if kind == "tab_line":
            ln = ev["line"]
            tag = "[mac]" if ev.get("kind") == "mac" else "[hermes]"
            role = ln.get("role")
            if role == "progress":
                text = ln["text"]
                print(f"{tag} {text if text.endswith('…') else text + '…'}", flush=True)
            elif role == "assistant":
                print(f"[celal] {ln['text']}", flush=True)
            elif role == "error":
                print(f"[hata] {ln['text']}", flush=True)
            elif role == "info":
                print(f"{tag} {ln['text']}", flush=True)
            elif role == "user" and ln.get("queued"):
                print(f"{tag} sırada (sekme meşgul ya da {settings.mac_max_parallel} iş çalışıyor)", flush=True)
        elif kind == "timing":
            print(ev["text"], flush=True)
        elif kind == "stopped":
            print(f"[durdur] {ev['text']}", flush=True)
        elif kind == "error":
            print(f"[hata] {ev['text']}", flush=True)
        elif kind == "info":
            print(f"[bilgi] {ev['text']}", flush=True)
        elif kind == "wake":
            print(f"[dinleme] {'aktif' if ev.get('active') else 'pasif'}: {ev['text']}", flush=True)

    voice = bool(args.mic)  # --text --mic: mic open too (speaker test / terminal voice chat)
    core = CelalCore(settings, cfg, voice=voice, emit=show)
    code = 0
    try:
        try:
            await core.start()
        except Exception as e:  # noqa: BLE001
            print(f"[hata] Celal başlatılamadı: {e}", flush=True)
            return 1
        active = core.tabs.active
        print(f"Beyin: Mac'teki Claude Code ({settings.mac_model}, {active.cwd if active else settings.mac_cwd}); "
              f"“sunucuda …” Hermes'e gider.", flush=True)
        if await core.wait_connected(args.connect_timeout):
            print(f"Hermes'e bağlı ({settings.api}, sohbet: {core.client.conversation}).", flush=True)
        else:
            print(f"[bilgi] Hermes'e bağlanılamadı: {core.reason} (Mac işleri yine çalışır)", flush=True)
        if core.mac_api is not None:
            print(f"Mac API: {core.mac_api_status} (Hermes → Mac, ters tünel ayrı ssh ile)", flush=True)
        if voice and cfg.wake:
            from celal.wake import START_PHRASE, STOP_PHRASE

            print(f"Mikrofon açık, dinleme pasif: “{START_PHRASE}” deyince dinler, “{STOP_PHRASE}” deyince durur.", flush=True)
        interactive = sys.stdin.isatty()
        lines = _stdin_lines()
        while True:
            if interactive:
                print("> ", end="", flush=True)
            line = await lines.get()
            if not line:
                break
            text = line.strip()
            if not text:
                continue
            if not interactive:
                print(f"[sen] {text}", flush=True)
            out = core.built.transport.output() if core.built else None
            bytes_before = getattr(out, "bytes_written", 0)
            tab = await core.send_text(text)
            if tab is None:
                code = 1
                continue
            before = tab.finished
            print(f"[{tab.kind}] sekme “{tab.label}” ({tab.cwd})", flush=True)
            deadline = time.monotonic() + args.timeout
            while tab.finished <= before and time.monotonic() < deadline and not tab.closed:
                await asyncio.sleep(0.1)
            if tab.finished <= before:
                print("[hata] cevap zaman aşımına uğradı", flush=True)
                code = 1
                continue
            if tab.status != "done":
                code = 1
            await core.wait_quiet(120)
            spoken = (getattr(out, "bytes_written", 0) - bytes_before) / (OUT_RATE * 2)
            print(f"[ses] {spoken:.1f} s okundu", flush=True)
    finally:
        await core.shutdown()
    if args.out:
        print(f"ses kaydı: {args.out}", flush=True)
    return code


DELEGATE_INSTRUCTIONS = (
    "Bu istek Mirza'nın Mac'indeki Claude Code'dan (Celal) geliyor: Mac'te yapılamayan, sunucuda yapılması gereken "
    "bir iş. İşi sunucuda yap ve sonucu eksiksiz, düz metin olarak yaz. İşi Mac'e (mac_task) geri gönderme."
)


async def _hermes_once(settings, text: str, timeout: float) -> int:
    """``celal hermes``: one request to Hermes; opens a temporary tunnel when none is up."""
    from celal.hermes import HermesClient, HermesError
    from celal.tunnel import TunnelManager

    client = HermesClient(settings, instructions=DELEGATE_INSTRUCTIONS, read_timeout=timeout)
    tunnel = None
    try:
        ok, reason = await client.ping()
        if not ok:
            if not settings.ssh_host:
                print(f"hata: Hermes'e ulaşılamadı ({reason}) ve [ssh] host ayarı yok", file=sys.stderr)
                return 2
            tunnel = TunnelManager(settings)
            tunnel.start()
            deadline = time.monotonic() + 30
            while not ok and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                ok, reason = await client.ping()
            if not ok:
                print(f"hata: Hermes'e bağlanılamadı: {reason} (tünel: {tunnel.status} {tunnel.detail})", file=sys.stderr)
                return 2
        wrote = False
        async for ev in client.stream(text):
            if ev.kind == "text":
                sys.stdout.write(ev.value)
                sys.stdout.flush()
                wrote = True
            elif ev.kind == "tool":
                print(f"[hermes aracı] {ev.value}", file=sys.stderr, flush=True)
        if wrote:
            sys.stdout.write("\n")
        else:
            print("(Hermes boş cevap verdi)", file=sys.stderr)
        return 0
    except HermesError as e:
        print(f"hata: {e}", file=sys.stderr)
        return 1
    finally:
        await client.aclose()
        if tunnel is not None:
            await tunnel.stop()


def run_hermes(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="celal hermes",
        description="Sunucudaki Hermes'e tek istek gönderir, cevabı stdout'a yazar (Mac'teki Claude bununla delege eder).",
    )
    parser.add_argument("text", nargs="*", help="istek (boşsa stdin)")
    parser.add_argument("--conversation", default=None, help="Hermes sohbet adı (varsayılan: <conversation>-claude)")
    parser.add_argument("--timeout", type=float, default=1200.0, help="cevap için en çok saniye")
    parser.add_argument("--log", default="warning")
    args = parser.parse_args(argv)
    _setup_logging(args.log)
    text = " ".join(args.text).strip() or sys.stdin.read().strip()
    if not text:
        print("hata: istek boş", file=sys.stderr)
        return 2
    from celal.config import load_settings

    settings = load_settings()
    settings.conversation = args.conversation or f"{settings.conversation}-claude"
    return asyncio.run(_hermes_once(settings, text, args.timeout))


def _stdin_lines() -> asyncio.Queue:
    """stdin lines via a daemon thread, so Ctrl-C never waits for a blocked readline."""
    import threading

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def reader():
        for line in sys.stdin:
            loop.call_soon_threadsafe(q.put_nowait, line)
        loop.call_soon_threadsafe(q.put_nowait, "")

    threading.Thread(target=reader, name="stdin", daemon=True).start()
    return q


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "hermes":
        return run_hermes(argv[1:])
    parser = argparse.ArgumentParser(
        prog="celal",
        description="Celal: Türkçe sesli asistan (beyin: Mac'teki Claude Code; sunucu işi: Hermes). "
        "Alt komut: celal hermes \"<istek>\"",
    )
    parser.add_argument("--text", action="store_true", help="terminal modu: stdin'den yazı → Mac Claude / Hermes → ses")
    parser.add_argument("--bench", action="store_true", help="Whisper / TTS / RAM ölçümü")
    from celal.voices import VOICES, WORKER_PACKAGES

    parser.add_argument("--tts", choices=[v.id for v in VOICES], default=None,
                        help="ses (varsayılan: pencerede son seçilen, yoksa ema)")
    parser.add_argument("--install-voice", choices=list(WORKER_PACKAGES),
                        help="ek ses motorunu Mac'in diskine kur (kitten: KittenTTS 2, ema-tts: büyük EMA)")
    parser.add_argument("--stt", choices=["turbo", "turbo-q4"], default="turbo", help="Whisper modeli")
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="Celal konuşurken araya girince sussun (kulaklık gerekir; varsayılan kapalı: Celal konuşurken mikrofon yok sayılır)",
    )
    parser.add_argument("--min-words", type=int, default=0, help="--barge-in ile: 0 sesini duyunca sus, N kelimeden sonra sus")
    parser.add_argument("--mic", action="store_true", help="mikrofon açık başlasın (pencere; --text ile terminalde de dinler)")
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help="sesli komut kapısı olmasın: mikrofon açıkken her söz Celal'e gider "
        "(varsayılan: “Celal dinlemeye başla” / “Celal dinlemeyi durdur” ile aç/kapat)",
    )
    parser.add_argument("--out", help="çalınan sesi WAV olarak da yaz")
    parser.add_argument("--no-play", action="store_true", help="hoparlöre çalma")
    parser.add_argument("--no-menubar", action="store_true", help="menü çubuğu simgesi olmasın")
    parser.add_argument("--timeout", type=float, default=300.0, help="--text: cevap için en çok saniye")
    parser.add_argument("--connect-timeout", type=float, default=30.0, help="--text: bağlantı için bekleme")
    parser.add_argument("--debug", action="store_true", help="pencerede web inspector")
    parser.add_argument("--log", default="warning", help="debug | info | warning")
    args = parser.parse_args(argv)
    _setup_logging(args.log)

    if args.install_voice:  # voices live on the Mac's disk, the model disk is not needed
        from celal.voices import install_worker

        return install_worker(args.install_voice)

    from celal.paths import ModelDiskMissing, configure_model_storage

    # Models default to the Mac's disk (celal.paths). Only a CELAL_MODELS_DIR
    # pointing at an unplugged disk fails; then the window and --text still
    # work: typing only, spoken with the chosen voice.
    disk_missing: str | None = None
    try:
        configure_model_storage()
    except ModelDiskMissing as e:
        if args.bench or args.mic:
            print(f"hata: {e}", file=sys.stderr)
            return 2
        disk_missing = str(e)
        print(f"uyarı: {e} Mikrofon kapalı, yalnız yazı.", file=sys.stderr)

    if args.bench:
        from celal.bench import run_bench

        return run_bench([args.stt], skip_ema=args.tts not in (None, "ema"))

    from celal.config import load_settings
    from celal.pipeline import Config
    from celal.voices import DEFAULT_VOICE, load_choice

    settings = load_settings()
    cfg = Config(
        tts=args.tts or load_choice(settings.state_dir) or DEFAULT_VOICE,
        stt=args.stt,
        min_words=args.min_words,
        barge_in=args.barge_in,
        play=not args.no_play,
        record_path=args.out,
        mic_on_start=args.mic,
        wake=not args.no_wake,
    )
    try:
        if args.text:
            return asyncio.run(_run_text(settings, cfg, args))
        from celal.ui import run_window

        notice = None
        if disk_missing:
            notice = ("CELAL_MODELS_DIR'daki disk takılı değil: mikrofon kapalı, yalnız yazı. Cevapları seçili ses "
                      "okumaya devam ediyor. Diski takıp ya da CELAL_MODELS_DIR'ı kaldırıp Celal'i yeniden aç.")
        return run_window(settings, cfg, debug=args.debug, menubar=not args.no_menubar,
                          voice=disk_missing is None, notice=notice)
    except KeyboardInterrupt:
        print("\nçıkılıyor", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
