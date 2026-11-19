import json
from pathlib import Path

import pytest

from caption_server.overwrite_policy import SubtitleProgress, decide_overwrite

CASES_FILE = Path(__file__).parents[2] / "test" / "fixtures" / "overwrite-policy-cases.json"
CASES = json.loads(CASES_FILE.read_text(encoding="utf-8"))


def _progress(raw: dict | None) -> SubtitleProgress | None:
    if raw is None:
        return None
    return SubtitleProgress(raw["status"], raw["coveredSeconds"], raw["sessionId"], raw["lastSeq"])


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_matches_shared_extension_cases(case: dict) -> None:
    decision = decide_overwrite(_progress(case["existing"]), _progress(case["candidate"]), case["confirmed"])
    expected = case["expected"]
    assert decision.write == expected["write"]
    assert decision.reason == expected.get("reason")
