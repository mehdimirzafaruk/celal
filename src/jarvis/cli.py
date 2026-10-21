"""Command line: ``jarvis run | text | bench``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from loguru import logger


def _setup_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


def _auth_note() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "uyarı: ANTHROPIC_API_KEY tanımlı; Claude Code abonelik yerine API anahtarını kullanabilir. "
            "Abonelik için: env -u ANTHROPIC_API_KEY uv run jarvis …",
            file=sys.stderr,
        )


async def _warm(built, stt_repo: str | None) -> None:
    """Load models before listening so the first turn is not slow."""
    t = time.perf_counter()
    tasks = [built.llm.warmup(), built.tts.warmup()]
    if stt_repo:

        async def warm_whisper():
            import mlx_whisper
            import numpy as np

            await asyncio.to_thread(
                mlx_whisper.transcribe, np.zeros(16000, dtype=np.float32), path_or_hf_repo=stt_repo, language="tr"
            )

        tasks.append(warm_whisper())
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    print(f"hazır ({time.perf_counter() - t:.1f}s ısınma)", flush=True)


async def _run_voice(args) -> int:
    from pipecat.workers.runner import WorkerRunner

    from jarvis.pipeline import STT_MODELS, Config, build_voice

    cfg = Config(
        tts=args.tts,
        stt=args.stt,
        model=args.model,
        min_words=args.min_words,
        play=not args.no_play,
        record_path=args.out,
        input_wav=args.wav,
    )
    built = build_voice(cfg)
    await _warm(built, STT_MODELS[cfg.stt])
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(built.worker)
    src = f"WAV {args.wav}" if args.wav else "mikrofon"
    print(f"Dinliyorum ({src}, tts={args.tts}, stt={args.stt}, model={args.model}). Ctrl-C ile çık.", flush=True)

    async def stopper():
        if args.duration:
            await asyncio.sleep(args.duration)
        elif args.wav:
            inp = built.transport.input()
            await inp.finished.wait()
            while not built.reply.done(settle=1.5) and built.reply.llm_started:
                await asyncio.sleep(0.2)
        else:
            return
        await built.worker.cancel(reason="duration elapsed")

    stop_task = asyncio.create_task(stopper())
    try:
        await runner.run()
    finally:
        stop_task.cancel()
        inp = built.transport.input()
        if hasattr(inp, "peak"):
            frames = getattr(inp, "frames_captured", 0)
            print(f"mikrofon: {frames} blok (20 ms) alındı, tepe genlik {inp.peak}/32767", flush=True)
            if frames and inp.peak == 0:
                print("uyarı: mikrofon hep sessiz; macOS mikrofon izni reddedilmiş olabilir.", flush=True)
    return 0


async def _run_text(args) -> int:
    from pipecat.frames.frames import LLMMessagesAppendFrame
    from pipecat.workers.runner import WorkerRunner

    from jarvis.pipeline import Config, build_text

    cfg = Config(tts=args.tts, model=args.model, play=not args.no_play, record_path=args.out)
    built = build_text(cfg)
    await _warm(built, None)
    started = asyncio.Event()

    @built.worker.event_handler("on_pipeline_started")
    async def _started(worker, frame):
        started.set()

    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(built.worker)
    run_task = asyncio.create_task(runner.run())
    await asyncio.wait_for(started.wait(), timeout=30)
    interactive = sys.stdin.isatty()
    try:
        while not run_task.done():
            if interactive:
                print("> ", end="", flush=True)
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            text = line.strip()
            if not text:
                continue
            if not interactive:
                print(f"[sen] {text}", flush=True)
            built.reply.reset()
            built.timer.prompt_submitted()
            await built.worker.queue_frame(
                LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True)
            )
            deadline = time.monotonic() + args.timeout
            while not built.reply.done() and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            if not built.reply.done():
                print("[hata] cevap zaman aşımına uğradı", flush=True)
    finally:
        await built.worker.end(reason="stdin closed")
        await run_task
    if args.out:
        print(f"ses kaydı: {args.out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="Vera: yerel Türkçe sesli asistan")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p, stt: bool = True):
        p.add_argument("--tts", choices=["ema", "say"], default="ema")
        if stt:
            p.add_argument("--stt", choices=["turbo", "turbo-q4"], default="turbo")
        p.add_argument("--model", default="haiku", help="haiku | sonnet | tam model id")
        p.add_argument("--log", default="warning", help="debug | info | warning")
        p.add_argument("--out", help="çalınan sesi WAV olarak da yaz")
        p.add_argument("--no-play", action="store_true", help="hoparlöre çalma (gerçek zamanlı akış sürer)")

    p_run = sub.add_parser("run", help="mikrofon → Vera → hoparlör")
    common(p_run)
    p_run.add_argument("--min-words", type=int, default=2, help="Vera konuşurken kesmek için en az kelime")
    p_run.add_argument("--duration", type=float, help="N saniye sonra kapan (test)")
    p_run.add_argument("--wav", help="mikrofon yerine bu WAV'ı çal (test)")

    p_text = sub.add_parser("text", help="stdin'den yazı → Vera → hoparlör")
    common(p_text, stt=False)
    p_text.add_argument("--timeout", type=float, default=120.0)

    p_bench = sub.add_parser("bench", help="Whisper / TTS / RAM ölçümü")
    p_bench.add_argument("--stt", choices=["turbo", "turbo-q4", "both"], default="both")
    p_bench.add_argument("--skip-ema", action="store_true")
    p_bench.add_argument("--log", default="warning")

    args = parser.parse_args(argv)
    _setup_logging(args.log)

    if args.cmd == "bench":
        from jarvis.bench import run_bench

        models = ["turbo", "turbo-q4"] if args.stt == "both" else [args.stt]
        return run_bench(models, skip_ema=args.skip_ema)

    _auth_note()
    try:
        if args.cmd == "run":
            return asyncio.run(_run_voice(args))
        return asyncio.run(_run_text(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
