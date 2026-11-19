import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from caption_server.stream_endpoint import FrameError, decode_audio_frame

from .conftest import ORIGIN, TOKEN

SR = 16_000
FRAME = 2048


def frame(start: float, samples: np.ndarray) -> bytes:
    pcm = np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes()
    return struct.pack("<d", start) + pcm


def speech_then_silence() -> np.ndarray:
    t = np.arange(int(1.5 * SR)) / SR
    return np.concatenate([(0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), np.zeros(SR, dtype=np.float32)])


def test_decode_audio_frame_round_trip_and_validation() -> None:
    start, samples = decode_audio_frame(frame(12.5, np.array([0.5, -0.5], dtype=np.float32)))
    assert start == 12.5
    assert samples.tolist() == pytest.approx([0.5, -0.5], abs=1e-4)
    for bad in [b"", b"12345678", struct.pack("<d", 1.0) + b"\x00", struct.pack("<d", -1.0) + b"\x00\x00", struct.pack("<d", float("nan")) + b"\x00\x00"]:
        with pytest.raises(FrameError):
            decode_audio_frame(bad)


def test_stream_rejects_wrong_origin(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/captions/stream", headers={"Origin": "https://evil.example"}) as ws:
            ws.receive_json()


def test_stream_rejects_wrong_token(client: TestClient) -> None:
    with client.websocket_connect("/v1/captions/stream", headers={"Origin": ORIGIN}) as ws:
        ws.send_json({"type": "start", "token": "wrong"})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_stream_returns_partial_final_and_coverage(client: TestClient, recognizer) -> None:
    recognizer.next_text = "第一句"
    audio = speech_then_silence()
    with client.websocket_connect("/v1/captions/stream", headers={"Origin": ORIGIN}) as ws:
        ws.send_json({"type": "start", "token": TOKEN})
        assert ws.receive_json() == {"type": "ready"}
        for offset in range(0, len(audio), FRAME):
            ws.send_bytes(frame(30.0 + offset / SR, audio[offset : offset + FRAME]))
        ws.send_json({"type": "stop"})
        messages = []
        try:
            while True:
                messages.append(ws.receive_json())
        except WebSocketDisconnect:
            pass
    kinds = {message["type"] for message in messages}
    assert {"final", "covered"} <= kinds
    final = next(message for message in messages if message["type"] == "final")
    assert final["start"] == pytest.approx(30.0, abs=0.05)
    assert final["segments"][0]["display"] == "第一句"


def test_stream_reports_bad_frames_without_closing(client: TestClient) -> None:
    with client.websocket_connect("/v1/captions/stream", headers={"Origin": ORIGIN}) as ws:
        ws.send_json({"type": "start", "token": TOKEN})
        ws.receive_json()
        ws.send_bytes(b"short")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "stop"})
