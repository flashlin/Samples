import io

import av
import numpy as np

TARGET_SAMPLE_RATE = 16_000


class AudioDecodeError(ValueError):
    pass


def decode_to_mono_16k(payload: bytes) -> np.ndarray:
    try:
        with av.open(io.BytesIO(payload)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioDecodeError("No audio stream found")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SAMPLE_RATE)
            parts = [
                resampled.to_ndarray().reshape(-1)
                for frame in container.decode(stream)
                for resampled in resampler.resample(frame)
            ]
            parts.extend(flushed.to_ndarray().reshape(-1) for flushed in resampler.resample(None))
    except (av.FFmpegError, StopIteration) as error:
        raise AudioDecodeError(f"Cannot decode audio: {error}") from error
    if not parts:
        raise AudioDecodeError("Audio stream is empty")
    return np.concatenate(parts).astype(np.float32, copy=False)
