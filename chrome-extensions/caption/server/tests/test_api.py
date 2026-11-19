import pytest
from fastapi.testclient import TestClient

from .conftest import TOKEN, encode_tone

KEY = "b" * 64


def put_body(covered: float, session: str = "s1", seq: int = 1, srt: str = "1\n00:00:00,000 --> 00:00:01,000\nhi\n", **extra) -> dict:
    return {
        "meta": {
            "canonicalUrl": "https://www.youtube.com/watch?v=x",
            "status": "partial",
            "coveredSeconds": covered,
            "coveredIntervals": [[0, covered]],
            "mediaDuration": 100,
            "sessionId": session,
            "lastSeq": seq,
        },
        "srt": srt,
        **extra,
    }


def test_health_reports_engines(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "engine": {"asr": "fake-asr", "translate": "fake-mt"}}


@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({"Authorization": ""}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Authorization": TOKEN}, 401),
        ({"Origin": "https://evil.example"}, 403),
        ({"Host": "evil.example:8765"}, 403),
        ({"Host": "127.0.0.1.evil.example"}, 403),
    ],
)
def test_guard_rejects_bad_requests(client: TestClient, headers: dict, expected_status: int) -> None:
    assert client.get("/health", headers=headers).status_code == expected_status


def test_requests_without_origin_are_allowed_with_token(client: TestClient) -> None:
    client.headers.pop("Origin")
    assert client.get("/health").status_code == 200


def test_transcribe_decodes_webm_and_returns_display_text(client: TestClient, recognizer) -> None:
    response = client.post("/v1/captions/transcribe", files={"audio": ("chunk.webm", encode_tone(1.5), "audio/webm")})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "這個是 decoder 的輸出"
    assert body["language"] == "zh"
    assert body["translation"] is None
    assert len(recognizer.received[0]) > 16_000


def test_transcribe_rejects_oversized_and_invalid_audio(client: TestClient) -> None:
    too_big = client.post("/v1/captions/transcribe", files={"audio": ("a.webm", b"0" * 200_001, "audio/webm")})
    assert too_big.status_code == 413
    invalid = client.post("/v1/captions/transcribe", files={"audio": ("a.webm", b"garbage", "audio/webm")})
    assert invalid.status_code == 422


def test_translations_return_bilingual_display_for_english_lines(client: TestClient) -> None:
    response = client.post("/v1/translations", json={"texts": ["Hello there friends", "中文句子"]})
    assert response.json() == {
        "translations": ["譯:Hello there friends", "中文句子"],
        "displays": ["Hello there friends\n譯:Hello there friends", "中文句子"],
    }


def test_subtitle_lifecycle(client: TestClient) -> None:
    assert client.get(f"/v1/subtitles/{KEY}").status_code == 404
    first = client.put(f"/v1/subtitles/{KEY}", json=put_body(10))
    assert first.json() == {"written": True, "revision": 1, "reason": None}
    document = client.get(f"/v1/subtitles/{KEY}").json()
    assert document["meta"]["coveredSeconds"] == 10
    assert document["meta"]["revision"] == 1
    assert document["meta"]["urlKey"] == KEY
    assert document["srt"].endswith("hi\n")
    behind = client.put(f"/v1/subtitles/{KEY}", json=put_body(5, session="other"))
    assert behind.json() == {"written": False, "revision": 1, "reason": "not-ahead"}
    conflict = client.put(f"/v1/subtitles/{KEY}", json=put_body(20, seq=2, expectedRevision=0))
    assert conflict.status_code == 409
    assert conflict.json()["revision"] == 1
    assert client.delete(f"/v1/subtitles/{KEY}").status_code == 204
    assert client.delete(f"/v1/subtitles/{KEY}").status_code == 404


@pytest.mark.parametrize("bad_key", ["short", "..%2F..%2Fetc", "G" * 64])
def test_subtitle_routes_reject_invalid_keys(client: TestClient, bad_key: str) -> None:
    assert client.get(f"/v1/subtitles/{bad_key}").status_code in {400, 404}
    assert client.put(f"/v1/subtitles/{bad_key}", json=put_body(1)).status_code in {400, 404}


def test_put_rejects_oversized_srt_and_unknown_fields(client: TestClient) -> None:
    assert client.put(f"/v1/subtitles/{KEY}", json=put_body(1, srt="x" * 1_001)).status_code == 413
    body = put_body(1)
    body["meta"]["unexpected"] = True
    assert client.put(f"/v1/subtitles/{KEY}", json=body).status_code == 422


def test_interim_transcription_skips_translation(client: TestClient, recognizer, translator) -> None:
    recognizer.next_text = "The transformer reads every token."
    files = {"audio": ("chunk.webm", encode_tone(1.0), "audio/webm")}
    interim = client.post("/v1/captions/transcribe", files=files, data={"translate": "false"}).json()
    assert (interim["language"], interim["translation"], interim["display"]) == ("en", None, "The transformer reads every token.")
    assert translator.calls == []
    final = client.post("/v1/captions/transcribe", files={"audio": ("chunk.webm", encode_tone(1.0), "audio/webm")}).json()
    assert final["display"] == "The transformer reads every token.\n譯:The transformer reads every token."
