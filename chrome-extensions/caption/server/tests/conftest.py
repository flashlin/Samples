from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from caption_server.app import CaptionServices, create_app
from caption_server.caption_pipeline import CaptionPipeline, PipelineOptions
from caption_server.engines import Recognition
from caption_server.sentence_segmenter import TimedToken
from caption_server.settings import Settings
from caption_server.subtitle_store import FileSubtitleStore

TOKEN = "test-token"
ORIGIN = "chrome-extension://abcdefghijklmnop"


class FakeRecognizer:
    name = "fake-asr"

    def __init__(self) -> None:
        self.next_text = "這个是 decoder 的输出"
        self.next_tokens: list[TimedToken] = []
        self.received: list[tuple[np.ndarray, bool]] = []

    def recognize(self, samples: np.ndarray, with_timestamps: bool) -> Recognition:
        self.received.append((samples, with_timestamps))
        return Recognition(text=self.next_text, tokens=self.next_tokens if with_timestamps else [])


class FakePunctuator:
    name = "fake-punct"

    def __init__(self) -> None:
        self.next_output: str | None = None

    def punctuate(self, text: str) -> str:
        return self.next_output if self.next_output is not None else text


class FakeTranslator:
    name = "fake-mt"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def translate(self, text: str, keep_terms: list[str]) -> str:
        self.calls.append((text, keep_terms))
        return f"译:{text}"


class _Segment:
    def __init__(self, start: int, samples: np.ndarray) -> None:
        self.start = start
        self.samples = samples


class EnergySpeechDetector:
    window = 512
    threshold = 0.05
    min_silence_windows = 4

    def __init__(self) -> None:
        self._position = 0
        self._speech_start: int | None = None
        self._speech: list[np.ndarray] = []
        self._silent_windows = 0
        self._finished: list[_Segment] = []

    def accept_waveform(self, samples: np.ndarray) -> None:
        for offset in range(0, len(samples), self.window):
            window = samples[offset : offset + self.window]
            loud = float(np.sqrt(np.mean(window**2))) > self.threshold
            if loud:
                if self._speech_start is None:
                    self._speech_start = self._position
                self._speech.append(window)
                self._silent_windows = 0
            elif self._speech_start is not None:
                self._speech.append(window)
                self._silent_windows += 1
                if self._silent_windows >= self.min_silence_windows:
                    self._close_segment()
            self._position += len(window)

    def _close_segment(self) -> None:
        if self._speech_start is not None:
            self._finished.append(_Segment(self._speech_start, np.concatenate(self._speech)))
        self._speech_start = None
        self._speech = []
        self._silent_windows = 0

    def is_speech_detected(self) -> bool:
        return self._speech_start is not None

    def empty(self) -> bool:
        return not self._finished

    @property
    def front(self) -> _Segment:
        return self._finished[0]

    def pop(self) -> None:
        self._finished.pop(0)

    def flush(self) -> None:
        self._close_segment()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, token=TOKEN, allowed_origins=[ORIGIN], max_srt_bytes=1_000)


@pytest.fixture
def recognizer() -> FakeRecognizer:
    return FakeRecognizer()


@pytest.fixture
def translator() -> FakeTranslator:
    return FakeTranslator()


@pytest.fixture
def punctuator() -> FakePunctuator:
    return FakePunctuator()


@pytest.fixture
def pipeline(recognizer: FakeRecognizer, punctuator: FakePunctuator, translator: FakeTranslator) -> CaptionPipeline:
    options = PipelineOptions(keep_terms=["transformer", "token"], bilingual_english=True, max_sentence_units=24)
    return CaptionPipeline(recognizer, punctuator, translator, options)


@pytest.fixture
def store(settings: Settings) -> FileSubtitleStore:
    return FileSubtitleStore(settings.subtitles_dir, settings.audit_log)


@pytest.fixture
def client(settings: Settings, pipeline: CaptionPipeline, store: FileSubtitleStore) -> TestClient:
    headers = {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN, "Host": f"127.0.0.1:{settings.port}"}
    services = CaptionServices(pipeline=pipeline, store=store, new_detector=EnergySpeechDetector)
    return TestClient(create_app(settings, services), base_url=f"http://127.0.0.1:{settings.port}", headers=headers)
