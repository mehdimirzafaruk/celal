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

    t_sent = [0.0]
    first_delta = [None]

    def show(ev: dict) -> None:
        kind = ev.get("type")
        if kind == "status":
            return
        if kind == "assistant_delta" and first_delta[0] is None:
            first_delta[0] = time.perf_counter()
        elif kind == "assistant_end":
            text = (ev.get("text") or "").strip()
            if text:
                print(f"[celal] {text}", flush=True)
            if t_sent[0] and first_delta[0]:
                print(
                    f"⏱ ilk kelime {first_delta[0] - t_sent[0]:.1f} s, tam cevap {time.perf_counter() - t_sent[0]:.1f} s",
                    flush=True,
                )
        elif kind == "tool":
            print(f"[araç] {ev['name']}", flush=True)
        elif kind == "error":
            print(f"[hata] {ev['text']}", flush=True)
        elif kind == "info":
            print(f"[bilgi] {ev['text']}", flush=True)
        elif kind == "user" and ev.get("source") == "mic":
            print(f"[sen (ses)] {ev['text']}", flush=True)
        elif kind == "mac":
            phase = ev.get("phase")
            if phase == "start":
                who = "Hermes'ten" if ev.get("source") == "hermes" else "Celal'den"
                print(f"[mac] {who} görev başladı ({ev.get('cwd')}): {ev.get('task')}", flush=True)
            elif phase == "progress":
                line = ev.get("text") or ""
                print(f"[mac] {line if line.endswith('…') else line + '…'}", flush=True)
            elif phase == "done":
                print(f"[mac] bitti: {ev.get('status')}, {ev.get('seconds')} s", flush=True)
                body = (ev.get("text") or "").strip()
                if body:
                    print(f"[mac sonuç]\n{body}", flush=True)
                if ev.get("summary") and ev.get("source") != "hermes":
                    print(f"[mac özet, sesli] {ev['summary']}", flush=True)

    cfg.print_sentences = False
    voice = bool(args.mic)  # --text --mic: mic open too (speaker test / terminal voice chat)
    core = CelalCore(settings, cfg, voice=voice, emit=show)
    code = 0
    try:
        # start() opens the ssh tunnel first; if it fails later (TTS, audio
        # device) the finally below still closes the tunnel.
        try:
            await core.start()
        except Exception as e:  # noqa: BLE001
            print(f"[hata] Celal başlatılamadı: {e}", flush=True)
            return 1
        if await core.wait_connected(args.connect_timeout):
            print(f"Hermes'e bağlı ({settings.api}, sohbet: {core.client.conversation}).", flush=True)
        else:
            print(f"[hata] Hermes'e bağlanılamadı: {core.reason} (tünel: {core.tunnel.status} {core.tunnel.detail})", flush=True)
        if core.mac_api is not None:
            print(f"Mac API: {core.mac_api_status} (Hermes → Mac, ters tünel ayrı ssh ile)", flush=True)
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
            t_sent[0], first_delta[0] = time.perf_counter(), None
            cut_before = core.interruptions
            out = core.built.transport.output() if core.built else None
            bytes_before = getattr(out, "bytes_written", 0)
            mac_req = core.mac_request(text)
            if mac_req is not None:
                task = await core.run_mac_task(mac_req[0], mac_req[1], shown=text)
                if task is None:
                    code = 1
                    continue
                if not await task.wait(settings.mac_timeout + 30):
                    print("[hata] Mac görevi zaman aşımına uğradı", flush=True)
                    code = 1
                    continue
                if task.status != "done":
                    code = 1
                try:
                    await asyncio.wait_for(core.speech_done.wait(), timeout=120)
                except TimeoutError:
                    print("[hata] özet okunamadı (zaman aşımı)", flush=True)
                spoken = (getattr(out, "bytes_written", 0) - bytes_before) / (OUT_RATE * 2)
                print(f"[ses] {spoken:.1f} s okundu", flush=True)
                continue
            if not await core.send_text(text):
                code = 1
                continue
            try:
                await asyncio.wait_for(core.turn_done.wait(), timeout=args.timeout)
            except TimeoutError:
                print("[hata] cevap zaman aşımına uğradı", flush=True)
                code = 1
            if core.last_error:
                code = 1
            spoken = (getattr(out, "bytes_written", 0) - bytes_before) / (OUT_RATE * 2)
            window = core.built.window if core.built else None
            gate = f", yankı kapısı {window.dropped_audio} mikrofon bloğu yok saydı" if window and voice else ""
            cut = core.interruptions - cut_before
            print(f"[ses] {spoken:.1f} s okundu, kesinti {cut}{gate}", flush=True)
    finally:
        await core.shutdown()
    if args.out:
        print(f"ses kaydı: {args.out}", flush=True)
    return code


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
    parser = argparse.ArgumentParser(prog="celal", description="Celal: Hermes'e bağlanan Türkçe sesli asistan")
    parser.add_argument("--text", action="store_true", help="terminal modu: stdin'den yazı → Hermes → ses")
    parser.add_argument("--bench", action="store_true", help="Whisper / TTS / RAM ölçümü")
    parser.add_argument("--tts", choices=["ema", "say"], default="ema", help="TTS motoru (varsayılan ema)")
    parser.add_argument("--stt", choices=["turbo", "turbo-q4"], default="turbo", help="Whisper modeli")
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="Celal konuşurken araya girince sussun (kulaklık gerekir; varsayılan kapalı: Celal konuşurken mikrofon yok sayılır)",
    )
    parser.add_argument("--min-words", type=int, default=0, help="--barge-in ile: 0 sesini duyunca sus, N kelimeden sonra sus")
    parser.add_argument("--mic", action="store_true", help="mikrofon açık başlasın (pencere; --text ile terminalde de dinler)")
    parser.add_argument("--out", help="çalınan sesi WAV olarak da yaz")
    parser.add_argument("--no-play", action="store_true", help="hoparlöre çalma")
    parser.add_argument("--no-menubar", action="store_true", help="menü çubuğu simgesi olmasın")
    parser.add_argument("--timeout", type=float, default=300.0, help="--text: cevap için en çok saniye")
    parser.add_argument("--connect-timeout", type=float, default=30.0, help="--text: bağlantı için bekleme")
    parser.add_argument("--debug", action="store_true", help="pencerede web inspector")
    parser.add_argument("--log", default="warning", help="debug | info | warning")
    args = parser.parse_args(argv)
    _setup_logging(args.log)

    from celal.paths import ModelDiskMissing, configure_model_storage

    # Window and --mic use Whisper; ema TTS and --bench load models too. Only
    # "--text --tts say" without a microphone runs without the model disk.
    needs_models = args.bench or not args.text or args.mic or args.tts == "ema"
    try:
        configure_model_storage()
    except ModelDiskMissing as e:
        if needs_models:
            print(f"hata: {e}", file=sys.stderr)
            return 2

    if args.bench:
        from celal.bench import run_bench

        return run_bench([args.stt], skip_ema=args.tts != "ema")

    from celal.config import load_settings
    from celal.pipeline import Config

    settings = load_settings()
    cfg = Config(
        tts=args.tts,
        stt=args.stt,
        min_words=args.min_words,
        barge_in=args.barge_in,
        play=not args.no_play,
        record_path=args.out,
        mic_on_start=args.mic,
    )
    try:
        if args.text:
            return asyncio.run(_run_text(settings, cfg, args))
        from celal.ui import run_window

        return run_window(settings, cfg, debug=args.debug, menubar=not args.no_menubar)
    except KeyboardInterrupt:
        print("\nçıkılıyor", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
