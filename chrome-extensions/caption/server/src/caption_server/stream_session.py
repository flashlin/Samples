import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .audio import TARGET_SAMPLE_RATE

Message = dict[str, Any]
Emit = Callable[[Message], Awaitable[None]]


class SpeechSegment(Protocol):
    start: int
    samples: Any


class SpeechDetector(Protocol):
    def accept_waveform(self, samples: np.ndarray) -> None: ...

    def is_speech_detected(self) -> bool: ...

    def empty(self) -> bool: ...

    @property
    def front(self) -> SpeechSegment: ...

    def pop(self) -> None: ...

    def flush(self) -> None: ...


@dataclass(frozen=True)
class TimedDisplay:
    start: float
    end: float
    display: str


@dataclass(frozen=True)
class StreamTuning:
    partial_interval_seconds: float = 0.5
    speech_lookback_seconds: float = 0.6
    covered_interval_seconds: float = 1.0
    discontinuity_seconds: float = 0.35
    max_buffer_seconds: float = 40.0


@dataclass(frozen=True)
class StreamEngines:
    new_detector: Callable[[], SpeechDetector]
    transcribe_partial: Callable[[np.ndarray], Awaitable[str]]
    transcribe_final: Callable[[np.ndarray], Awaitable[list[TimedDisplay]]]


@dataclass(frozen=True)
class _FinalJob:
    start: float
    samples: np.ndarray


@dataclass(frozen=True)
class _CoveredJob:
    start: float
    end: float


@dataclass(frozen=True)
class _PartialJob:
    start: float
    end: float
    samples: np.ndarray


def _seconds(samples: int) -> float:
    return samples / TARGET_SAMPLE_RATE


def _samples(seconds: float) -> int:
    return int(round(seconds * TARGET_SAMPLE_RATE))


class CaptionStreamSession:
    def __init__(self, engines: StreamEngines, emit: Emit, tuning: StreamTuning) -> None:
        self._engines = engines
        self._emit = emit
        self._tuning = tuning
        self._jobs: deque[_FinalJob | _CoveredJob] = deque()
        self._pending_partial: _PartialJob | None = None
        self._wake = asyncio.Event()
        self._closing = False
        self._last_final_end = float("-inf")
        self._worker = asyncio.create_task(self._work())
        self._reset_run()

    def _reset_run(self) -> None:
        self._detector = self._engines.new_detector()
        self._origin: float | None = None
        self._received = 0
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_offset = 0
        self._speech_start: int | None = None
        self._last_partial_at = 0
        self._covered_until = 0

    def _time(self, sample_index: int) -> float:
        return (self._origin or 0.0) + _seconds(sample_index)

    async def feed(self, start: float, samples: np.ndarray) -> None:
        if self._origin is not None and abs(start - self._time(self._received)) > self._tuning.discontinuity_seconds:
            await self.boundary()
        if self._origin is None:
            self._origin = start
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        self._received += len(samples)
        self._detector.accept_waveform(samples)
        self._collect_finished_speech()
        self._track_speech_in_progress()
        self._report_covered_silence()
        self._trim_buffer()
        self._wake.set()

    async def boundary(self) -> None:
        if self._origin is None:
            return
        self._detector.flush()
        self._collect_finished_speech()
        self._speech_start = None
        self._report_covered_silence(force=True)
        self._reset_run()
        self._wake.set()

    async def close(self) -> None:
        await self.boundary()
        self._closing = True
        self._wake.set()
        await self._worker

    def _collect_finished_speech(self) -> None:
        while not self._detector.empty():
            segment = self._detector.front
            samples = np.asarray(segment.samples, dtype=np.float32)
            self._detector.pop()
            if segment.start > self._covered_until:
                self._jobs.append(_CoveredJob(start=self._time(self._covered_until), end=self._time(segment.start)))
            self._jobs.append(_FinalJob(start=self._time(segment.start), samples=samples))
            self._covered_until = max(self._covered_until, segment.start + len(samples))
            self._speech_start = None
            self._pending_partial = None

    def _track_speech_in_progress(self) -> None:
        if not self._detector.is_speech_detected():
            self._speech_start = None
            return
        if self._speech_start is None:
            self._speech_start = max(self._covered_until, self._received - _samples(self._tuning.speech_lookback_seconds), self._buffer_offset)
            self._last_partial_at = self._speech_start
            self._report_covered_silence(force=True)
        if self._received - self._last_partial_at >= _samples(self._tuning.partial_interval_seconds):
            self._last_partial_at = self._received
            begin = self._speech_start - self._buffer_offset
            audio = self._buffer[begin:].copy()
            self._pending_partial = _PartialJob(start=self._time(self._speech_start), end=self._time(self._received), samples=audio)

    def _report_covered_silence(self, force: bool = False) -> None:
        until = self._speech_start if self._speech_start is not None else self._received
        if until - self._covered_until >= _samples(self._tuning.covered_interval_seconds) or (force and until > self._covered_until):
            self._jobs.append(_CoveredJob(start=self._time(self._covered_until), end=self._time(until)))
            self._covered_until = until

    def _trim_buffer(self) -> None:
        keep_from = self._speech_start if self._speech_start is not None else self._received - _samples(self._tuning.speech_lookback_seconds)
        keep_from = max(keep_from, self._received - _samples(self._tuning.max_buffer_seconds))
        drop = keep_from - self._buffer_offset
        if drop > 0:
            self._buffer = self._buffer[drop:]
            self._buffer_offset = keep_from

    def _next_job(self) -> _FinalJob | _CoveredJob | _PartialJob | None:
        if self._jobs:
            return self._jobs.popleft()
        partial, self._pending_partial = self._pending_partial, None
        return partial

    async def _work(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                if self._closing:
                    return
                self._wake.clear()
                await self._wake.wait()
                continue
            try:
                await self._run(job)
            except Exception as error:
                await self._emit({"type": "error", "message": f"Recognition failed: {error}"})

    async def _run(self, job: _FinalJob | _CoveredJob | _PartialJob) -> None:
        if isinstance(job, _CoveredJob):
            await self._emit({"type": "covered", "start": job.start, "end": job.end})
        elif isinstance(job, _FinalJob):
            await self._run_final(job)
        elif job.end > self._last_final_end:
            text = await self._engines.transcribe_partial(job.samples)
            if text and job.end > self._last_final_end:
                await self._emit({"type": "partial", "start": job.start, "end": job.end, "text": text})

    async def _run_final(self, job: _FinalJob) -> None:
        end = job.start + _seconds(len(job.samples))
        self._last_final_end = max(self._last_final_end, end)
        captions = await self._engines.transcribe_final(job.samples)
        segments = [
            {"start": job.start + caption.start, "end": min(end, job.start + caption.end), "display": caption.display}
            for caption in captions
            if caption.display
        ]
        await self._emit({"type": "final", "start": job.start, "end": end, "segments": segments})
