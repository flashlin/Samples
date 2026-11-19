import io
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from caption_server.app import create_app
from caption_server.caption_pipeline import CaptionPipeline
from caption_server.settings import Settings
from caption_server.subtitle_store import FileSubtitleStore

TOKEN = "test-token"
ORIGIN = "chrome-extension://abcdefghijklmnop"
SAMPLE_RATE = 48_000


class FakeRecognizer:
    name = "fake-asr"

    def __init__(self) -> None:
        self.next_text = "這个是 decoder 的输出"
        self.received: list[np.ndarray] = []

    def transcribe(self, samples: np.ndarray) -> str:
        self.received.append(samples)
        return self.next_text


class FakeTranslator:
    name = "fake-mt"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def translate(self, text: str, keep_terms: list[str]) -> str:
        self.calls.append((text, keep_terms))
        return f"译:{text}"


def encode_tone(seconds: float, container_format: str = "webm", codec: str = "libopus") -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=SAMPLE_RATE, layout="mono")
        samples = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, token=TOKEN, allowed_origins=[ORIGIN], max_audio_bytes=200_000, max_srt_bytes=1_000)


@pytest.fixture
def recognizer() -> FakeRecognizer:
    return FakeRecognizer()


@pytest.fixture
def translator() -> FakeTranslator:
    return FakeTranslator()


@pytest.fixture
def pipeline(recognizer: FakeRecognizer, translator: FakeTranslator) -> CaptionPipeline:
    return CaptionPipeline(recognizer, translator, keep_terms=["transformer", "token"], bilingual_english=True)


@pytest.fixture
def store(settings: Settings) -> FileSubtitleStore:
    return FileSubtitleStore(settings.subtitles_dir, settings.audit_log)


@pytest.fixture
def client(settings: Settings, pipeline: CaptionPipeline, store: FileSubtitleStore) -> TestClient:
    headers = {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN, "Host": f"127.0.0.1:{settings.port}"}
    return TestClient(create_app(settings, pipeline, store), base_url=f"http://127.0.0.1:{settings.port}", headers=headers)
