import pytest

from caption_server.audio_decoder import TARGET_SAMPLE_RATE, AudioDecodeError, decode_to_mono_16k

from .conftest import encode_tone


@pytest.mark.parametrize(("container_format", "codec"), [("webm", "libopus"), ("wav", "pcm_s16le")])
def test_decodes_recorder_formats_to_mono_16k(container_format: str, codec: str) -> None:
    samples = decode_to_mono_16k(encode_tone(2.0, container_format, codec))
    assert samples.ndim == 1
    assert samples.dtype.name == "float32"
    assert abs(len(samples) / TARGET_SAMPLE_RATE - 2.0) < 0.1
    assert samples.max() > 0.1


@pytest.mark.parametrize("payload", [b"", b"not audio at all", b"\x1aE\xdf\xa3broken"])
def test_rejects_invalid_payloads(payload: bytes) -> None:
    with pytest.raises(AudioDecodeError):
        decode_to_mono_16k(payload)
