"""Local audio transports for Pipecat built on ``sounddevice``.

Pipecat's ``LocalAudioTransport`` needs PyAudio, which on macOS needs
``brew install portaudio`` to build. ``sounddevice``'s macOS wheel bundles
PortAudio, so this module mirrors ``pipecat.transports.local.audio`` with the
same structure (BaseInputTransport / BaseOutputTransport / BaseTransport) but
no system dependency.

Extras used by the CLI:
- ``play=False`` on output: frames are paced in real time but not played
  (bot speaking events still fire), for headless runs.
- ``record_path`` on output: everything that would be played is also written
  to a WAV file (post-interruption, at the transport sample rate).
- ``WavFileInputTransport``: feeds a WAV file as if it were the microphone,
  followed by silence, to exercise VAD → turn → STT without a mic.
"""

from __future__ import annotations

import asyncio
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd
from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams


class SoundDeviceTransportParams(TransportParams):
    """Transport params plus device selection and headless/recording options."""

    input_device: int | str | None = None
    output_device: int | str | None = None
    play: bool = True
    record_path: str | None = None
    input_wav: str | None = None
    input_wav_tail_secs: float = 8.0
    capture_on_start: bool = True


class SoundDeviceInputTransport(BaseInputTransport):
    """Microphone capture (int16, 20 ms blocks) → ``InputAudioRawFrame``."""

    _params: SoundDeviceTransportParams

    def __init__(self, params: SoundDeviceTransportParams):
        super().__init__(params)
        self._stream: sd.RawInputStream | None = None
        self.frames_captured = 0
        self.peak = 0  # max |sample| seen; stays 0 when macOS denies mic access
        self.capturing = params.capture_on_start

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        blocksize = int(self.sample_rate / 100) * 2  # 20 ms
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self._params.audio_in_channels,
            dtype="int16",
            blocksize=blocksize,
            device=self._params.input_device,
            callback=self._audio_in_callback,
        )

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._stream:
            if self.capturing:
                self._stream.start()
            await self.set_transport_ready(frame)

    def set_capturing(self, on: bool) -> None:
        """Mic button: open/close the input stream (macOS mic indicator follows)."""
        self.capturing = on
        if not self._stream:
            return
        if on and not self._stream.active:
            self._stream.start()
        elif not on and self._stream.active:
            self._stream.stop()

    async def cleanup(self):
        await super().cleanup()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"input stream close: {e}")
            self._stream = None

    def _audio_in_callback(self, indata, frames, time_info, status):
        if not self.capturing:
            return
        if status:
            logger.debug(f"input status: {status}")
        self.frames_captured += 1
        data = bytes(indata)
        if data:
            self.peak = max(self.peak, int(np.abs(np.frombuffer(data, dtype="<i2")).max()))
        frame = InputAudioRawFrame(
            audio=data,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_in_channels,
        )
        asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), self.get_event_loop())


def read_wav_mono16(path: str | Path, target_rate: int) -> bytes:
    """Read a WAV as mono int16 PCM at ``target_rate`` (linear resample if needed)."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit WAV is supported")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        n = int(len(audio) * target_rate / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return np.clip(audio, -32768, 32767).astype("<i2").tobytes()


class WavFileInputTransport(BaseInputTransport):
    """Plays a WAV file into the pipeline in real time, then silence."""

    _params: SoundDeviceTransportParams

    def __init__(self, params: SoundDeviceTransportParams):
        super().__init__(params)
        self._feed_task: asyncio.Task | None = None
        self.finished = asyncio.Event()
        self.capturing = params.capture_on_start

    def set_capturing(self, on: bool) -> None:
        """Mic button in headless runs: only the flag changes, the file keeps playing."""
        self.capturing = on

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        if not self._feed_task:
            self._feed_task = self.create_task(self._feed(), "wav-feed")

    async def cleanup(self):
        await super().cleanup()
        if self._feed_task:
            await self.cancel_task(self._feed_task)
            self._feed_task = None

    async def _feed(self):
        assert self._params.input_wav
        pcm = read_wav_mono16(self._params.input_wav, self.sample_rate)
        step = int(self.sample_rate / 100) * 2 * 2  # 20 ms of int16
        silence = b"\x00" * step
        tail = int(self._params.input_wav_tail_secs / 0.02)
        start = time.monotonic()
        n = 0
        chunks = [pcm[i : i + step] for i in range(0, len(pcm), step)] + [silence] * tail
        for chunk in chunks:
            if len(chunk) < step:
                chunk = chunk + b"\x00" * (step - len(chunk))
            await self.push_audio_frame(
                InputAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=1)
            )
            n += 1
            delay = start + n * 0.02 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        self.finished.set()


class SoundDeviceOutputTransport(BaseOutputTransport):
    """Speaker playback of ``OutputAudioRawFrame`` (optionally headless and/or recorded)."""

    _params: SoundDeviceTransportParams

    def __init__(self, params: SoundDeviceTransportParams):
        super().__init__(params)
        self._stream: sd.RawOutputStream | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._wav: wave.Wave_write | None = None
        self.bytes_written = 0
        self._clock_start: float | None = None
        self._clock_bytes = 0

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        if self._params.play:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self._params.audio_out_channels,
                dtype="int16",
                device=self._params.output_device,
            )
        if self._params.record_path:
            self._wav = wave.open(self._params.record_path, "wb")
            self._wav.setnchannels(self._params.audio_out_channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(self.sample_rate)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._stream:
            self._stream.start()
        await self.set_transport_ready(frame)

    async def cleanup(self):
        try:
            await super().cleanup()
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"output stream close: {e}")
                self._stream = None
            self.close_recording()
        finally:
            self._executor.shutdown(wait=False)

    def close_recording(self) -> None:
        if self._wav:
            self._wav.close()
            self._wav = None

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._wav:
            self._wav.writeframes(frame.audio)
        self.bytes_written += len(frame.audio)
        if self._stream:
            await self.get_event_loop().run_in_executor(self._executor, self._stream.write, frame.audio)
            return True
        # Headless: pace like a real device so speaking/turn timing stays realistic.
        bytes_per_sec = self.sample_rate * 2 * self._params.audio_out_channels
        now = time.monotonic()
        if self._clock_start is None or now - (self._clock_start + self._clock_bytes / bytes_per_sec) > 0.2:
            self._clock_start, self._clock_bytes = now, 0
        self._clock_bytes += len(frame.audio)
        delay = self._clock_start + self._clock_bytes / bytes_per_sec - now
        if delay > 0:
            await asyncio.sleep(delay)
        return True


class SoundDeviceTransport(BaseTransport):
    """Input + output local audio transport (drop-in for ``LocalAudioTransport``)."""

    def __init__(self, params: SoundDeviceTransportParams):
        super().__init__()
        self._params = params
        self._input: BaseInputTransport | None = None
        self._output: SoundDeviceOutputTransport | None = None

    def input(self) -> FrameProcessor:
        if not self._input:
            if self._params.input_wav:
                self._input = WavFileInputTransport(self._params)
            else:
                self._input = SoundDeviceInputTransport(self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = SoundDeviceOutputTransport(self._params)
        return self._output
