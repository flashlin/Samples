import pytest
from fastapi.testclient import TestClient

from .conftest import TOKEN

KEY = "b" * 64


def put_body(covered: float, session: str = "s1", seq: int = 1, text: str = "hi", **extra) -> dict:
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
        "cues": [{"start": 0.0, "end": 1.0, "text": text}],
        **extra,
    }


def test_health_reports_engines(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "engine": {"asr": "fake-asr", "punctuation": "fake-punct", "translate": "fake-mt"}}


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
    other = client.put(f"/v1/subtitles/{KEY}", json=put_body(50, session="other"))
    assert other.json() == {"written": False, "revision": 1, "reason": "confirmation-required"}
    confirmed = client.put(f"/v1/subtitles/{KEY}", json=put_body(5, session="other", confirmOverwrite=True, expectedRevision=1))
    assert confirmed.json() == {"written": True, "revision": 2, "reason": None}
    conflict = client.put(f"/v1/subtitles/{KEY}", json=put_body(20, session="other", seq=2, expectedRevision=0))
    assert conflict.status_code == 409
    assert conflict.json()["revision"] == 2
    assert client.delete(f"/v1/subtitles/{KEY}").status_code == 204
    assert client.delete(f"/v1/subtitles/{KEY}").status_code == 404


@pytest.mark.parametrize("bad_key", ["short", "..%2F..%2Fetc", "G" * 64])
def test_subtitle_routes_reject_invalid_keys(client: TestClient, bad_key: str) -> None:
    assert client.get(f"/v1/subtitles/{bad_key}").status_code in {400, 404}
    assert client.put(f"/v1/subtitles/{bad_key}", json=put_body(1)).status_code in {400, 404}


def test_put_rejects_oversized_cues_and_unknown_fields(client: TestClient) -> None:
    assert client.put(f"/v1/subtitles/{KEY}", json=put_body(1, text="x" * 2_001)).status_code == 422
    body = put_body(1)
    body["meta"]["unexpected"] = True
    assert client.put(f"/v1/subtitles/{KEY}", json=body).status_code == 422




def test_undo_endpoint_restores_the_previous_recording(client: TestClient) -> None:
    client.put(f"/v1/subtitles/{KEY}", json=put_body(10, text="first"))
    client.put(f"/v1/subtitles/{KEY}", json=put_body(10, session="other", text="second", confirmOverwrite=True))
    undone = client.post(f"/v1/subtitles/{KEY}/undo").json()
    assert undone["restored"] is True
    assert client.get(f"/v1/subtitles/{KEY}").json()["srt"].endswith("first\n")
