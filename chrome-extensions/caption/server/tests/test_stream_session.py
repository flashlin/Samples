import asyncio

import numpy as np
import pytest

from caption_server.stream_session import CaptionStreamSession, StreamEngines, StreamTuning, TimedDisplay

from .conftest import EnergySpeechDetector

SR = 16_000
FRAME = 2048


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


class Recorder:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.partial_calls = 0
        self.final_lengths: list[float] = []

    async def emit(self, message: dict) -> None:
        self.messages.append(message)

    async def partial(self, samples: np.ndarray) -> str:
        self.partial_calls += 1
        await asyncio.sleep(0)
        return f"partial {len(samples) / SR:.1f}s"

    async def final(self, samples: np.ndarray) -> list[TimedDisplay]:
        seconds = len(samples) / SR
        self.final_lengths.append(seconds)
        return [TimedDisplay(0.0, seconds / 2, "第一句。"), TimedDisplay(seconds / 2, seconds, "第二句？")]

    def of_type(self, kind: str) -> list[dict]:
        return [message for message in self.messages if message["type"] == kind]


def make_session(recorder: Recorder) -> CaptionStreamSession:
    engines = StreamEngines(new_detector=EnergySpeechDetector, transcribe_partial=recorder.partial, transcribe_final=recorder.final)
    return CaptionStreamSession(engines, recorder.emit, StreamTuning(partial_interval_seconds=0.5, covered_interval_seconds=1.0))


async def stream(session: CaptionStreamSession, audio: np.ndarray, start: float) -> None:
    for offset in range(0, len(audio), FRAME):
        await session.feed(start + offset / SR, audio[offset : offset + FRAME])
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_speech_produces_partials_then_a_timed_final() -> None:
    recorder = Recorder()
    session = make_session(recorder)
    await stream(session, np.concatenate([silence(1.0), tone(2.0), silence(1.0)]), start=10.0)
    await session.close()
    finals = recorder.of_type("final")
    assert len(finals) == 1
    final = finals[0]
    assert final["start"] == pytest.approx(11.0, abs=0.05)
    assert [segment["display"] for segment in final["segments"]] == ["第一句。", "第二句？"]
    assert final["segments"][0]["start"] == pytest.approx(final["start"])
    assert recorder.partial_calls >= 2
    assert all(10.0 <= message["start"] <= final["start"] for message in recorder.of_type("partial"))


@pytest.mark.asyncio
async def test_coverage_reports_silence_and_speech_without_gaps() -> None:
    recorder = Recorder()
    session = make_session(recorder)
    await stream(session, np.concatenate([silence(2.0), tone(1.5), silence(2.0)]), start=0.0)
    await session.close()
    ranges = sorted((m["start"], m["end"]) for m in recorder.of_type("covered")) + [(f["start"], f["end"]) for f in recorder.of_type("final")]
    ranges.sort()
    merged_end = 0.0
    for start, end in ranges:
        assert start <= merged_end + 0.05
        merged_end = max(merged_end, end)
    assert merged_end == pytest.approx(5.5, abs=0.15)


@pytest.mark.asyncio
async def test_break_finishes_the_current_sentence_and_restarts_the_timeline() -> None:
    recorder = Recorder()
    session = make_session(recorder)
    await stream(session, tone(1.5), start=0.0)
    await session.boundary()
    await stream(session, np.concatenate([tone(1.0), silence(1.0)]), start=100.0)
    await session.close()
    starts = [round(final["start"]) for final in recorder.of_type("final")]
    assert starts == [0, 100]


@pytest.mark.asyncio
async def test_time_jump_is_treated_as_a_break() -> None:
    recorder = Recorder()
    session = make_session(recorder)
    await stream(session, tone(1.0), start=0.0)
    await stream(session, np.concatenate([tone(1.0), silence(1.0)]), start=50.0)
    await session.close()
    assert [round(final["start"]) for final in recorder.of_type("final")] == [0, 50]


@pytest.mark.asyncio
async def test_silence_only_never_calls_the_recognizer() -> None:
    recorder = Recorder()
    session = make_session(recorder)
    await stream(session, silence(3.0), start=0.0)
    await session.close()
    assert recorder.partial_calls == 0
    assert recorder.final_lengths == []
    assert recorder.of_type("covered")


@pytest.mark.asyncio
async def test_recognizer_failure_is_reported_and_the_stream_continues() -> None:
    recorder = Recorder()
    calls = {"count": 0}

    async def flaky_final(samples: np.ndarray) -> list[TimedDisplay]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")
        return [TimedDisplay(0.0, 0.5, "ok")]

    engines = StreamEngines(new_detector=EnergySpeechDetector, transcribe_partial=recorder.partial, transcribe_final=flaky_final)
    session = CaptionStreamSession(engines, recorder.emit, StreamTuning())
    await stream(session, np.concatenate([tone(1.0), silence(1.0), tone(1.0), silence(1.0)]), start=0.0)
    await session.close()
    assert recorder.of_type("error")[0]["message"].endswith("boom")
    assert [segment["display"] for final in recorder.of_type("final") for segment in final["segments"]] == ["ok"]
