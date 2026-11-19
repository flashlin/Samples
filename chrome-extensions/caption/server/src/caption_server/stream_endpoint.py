import asyncio
import json
import struct
from collections.abc import Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect, status

from .caption_pipeline import CaptionPipeline
from .security import RequestGuard
from .stream_session import CaptionStreamSession, Message, SpeechDetector, StreamEngines, StreamTuning, TimedDisplay

FRAME_HEADER = struct.Struct("<d")
INT16_SCALE = 32768.0
MAX_FRAME_BYTES = 256_000
HELLO_TIMEOUT_SECONDS = 10


class FrameError(ValueError):
    pass


def decode_audio_frame(payload: bytes) -> tuple[float, np.ndarray]:
    if len(payload) <= FRAME_HEADER.size or len(payload) > MAX_FRAME_BYTES or (len(payload) - FRAME_HEADER.size) % 2:
        raise FrameError("Audio frame must be an 8-byte start time followed by 16-bit PCM samples")
    (start,) = FRAME_HEADER.unpack_from(payload)
    if not np.isfinite(start) or start < 0:
        raise FrameError("Audio frame start time must be a non-negative number")
    samples = np.frombuffer(payload, dtype="<i2", offset=FRAME_HEADER.size).astype(np.float32) / INT16_SCALE
    return start, samples


def pipeline_engines(pipeline: CaptionPipeline, new_detector: Callable[[], SpeechDetector]) -> StreamEngines:
    async def transcribe_partial(samples: np.ndarray) -> str:
        chunk = await asyncio.to_thread(pipeline.transcribe, samples, False)
        return chunk.caption.display

    async def transcribe_final(samples: np.ndarray) -> list[TimedDisplay]:
        chunk = await asyncio.to_thread(pipeline.transcribe, samples, True)
        return [TimedDisplay(segment.start, segment.end, segment.caption.display) for segment in chunk.segments]

    return StreamEngines(new_detector=new_detector, transcribe_partial=transcribe_partial, transcribe_final=transcribe_final)


async def _authenticate(websocket: WebSocket, guard: RequestGuard) -> bool:
    if not guard.accepts_websocket(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False
    await websocket.accept()
    try:
        hello = await asyncio.wait_for(websocket.receive_json(), HELLO_TIMEOUT_SECONDS)
    except (TimeoutError, ValueError, WebSocketDisconnect):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False
    if not isinstance(hello, dict) or hello.get("type") != "start" or not guard.is_valid_token(hello.get("token")):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False
    return True


async def serve_caption_stream(websocket: WebSocket, guard: RequestGuard, engines: StreamEngines, tuning: StreamTuning) -> None:
    if not await _authenticate(websocket, guard):
        return

    async def emit(message: Message) -> None:
        try:
            await websocket.send_json(message)
        except (WebSocketDisconnect, RuntimeError):
            return

    session = CaptionStreamSession(engines, emit, tuning)
    await emit({"type": "ready"})
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                try:
                    start, samples = decode_audio_frame(message["bytes"])
                except FrameError as error:
                    await emit({"type": "error", "message": str(error)})
                    continue
                await session.feed(start, samples)
            elif message.get("text") is not None:
                command = json.loads(message["text"])
                if command.get("type") == "break":
                    await session.boundary()
                elif command.get("type") == "stop":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass
