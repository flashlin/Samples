import argparse
import json
from pathlib import Path

import pytest

import video_gen_cli
from pi_h3_assets import adapter_name
from pi_h3_gen import PiH3Profile
from sulphur_gen import SulphurProfile
from video_gen_cli import (
    MODES,
    RunSettings,
    build_segment_workflow,
    continuation_frame_name,
    final_output_path,
    first_frame_name,
    list_prompt_choices,
    split_into_segments,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TIMESTAMP = "20260101_000000"


def make_settings(profile, mode_key: str, args: argparse.Namespace, steps: int) -> RunSettings:
    return RunSettings(
        profile=profile,
        mode=next(m for m in MODES if m.key == mode_key),
        prompt="PROMPT",
        width=672,
        height=384,
        segment_seconds=[10.0, 10.0],
        seed=7,
        steps=steps,
        timestamp=TIMESTAMP,
        args=args,
    )


def linked_node_ids(workflow: dict) -> set:
    return {
        value[0]
        for node in workflow.values()
        for value in node["inputs"].values()
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
    }


@pytest.mark.parametrize("mode_key", ["t2v", "i2v"])
@pytest.mark.parametrize("corrector", [True, False])
@pytest.mark.parametrize("index", [1, 2])
def test_sulphur_workflow_matches_pre_refactor_snapshot(mode_key, corrector, index):
    settings = make_settings(SulphurProfile(), mode_key, argparse.Namespace(no_corrector=not corrector), steps=20)
    first_frame = "_s3_20260101_000000_first_frame.png" if mode_key == "i2v" else None
    frame = first_frame if index == 1 else continuation_frame_name(settings, 1)
    fixture = FIXTURES_DIR / f"sulphur3_{mode_key}_{'corr' if corrector else 'nocorr'}_seg{index}.json"
    assert build_segment_workflow(settings, index, frame) == json.loads(fixture.read_text(encoding="utf-8"))


def test_sulphur_file_names_keep_s3_convention():
    settings = make_settings(SulphurProfile(), "t2v", argparse.Namespace(no_corrector=False), steps=20)
    assert continuation_frame_name(settings, 1) == "_s3_20260101_000000_last_frame_01.png"
    assert first_frame_name(settings, ".png") == "_s3_20260101_000000_first_frame.png"
    assert final_output_path(settings).name == "sulphur3_t2v_672x384_20s_20260101_000000.mp4"


@pytest.mark.parametrize("steps, max_raw_t", [(2, 0.93), (4, 0.96)])
def test_pi_h3_text_to_video_uses_adapter_for_steps(steps, max_raw_t):
    settings = make_settings(PiH3Profile(), "t2v", argparse.Namespace(), steps=steps)
    workflow = build_segment_workflow(settings, 1, None)
    assert workflow["39"]["inputs"]["adapter_name"] == adapter_name(steps)
    assert workflow["8"]["inputs"]["steps"] == steps
    assert workflow["8"]["inputs"]["max_raw_t"] == max_raw_t
    assert workflow["6"]["inputs"]["length"] == 243
    assert "first_frame" not in workflow["6"]["inputs"]
    assert "21" not in workflow
    assert workflow["15"]["inputs"]["filename_prefix"] == "pih3/20260101_000000_t2v_seg01"
    assert linked_node_ids(workflow) <= set(workflow)


def test_pi_h3_continuation_scales_first_frame_to_canvas():
    settings = make_settings(PiH3Profile(), "t2v", argparse.Namespace(), steps=4)
    frame = continuation_frame_name(settings, 1)
    workflow = build_segment_workflow(settings, 2, frame)
    assert frame == "_pih3_20260101_000000_last_frame_01.png"
    assert workflow["21"]["inputs"]["image"] == frame
    assert workflow["23"]["inputs"]["width"] == 672
    assert workflow["23"]["inputs"]["height"] == 384
    assert workflow["6"]["inputs"]["first_frame"] == ["23", 0]
    assert workflow["15"]["inputs"]["filename_prefix"] == "pih3/20260101_000000_t2v_seg02"
    assert linked_node_ids(workflow) <= set(workflow)


def test_prompt_choices_put_default_then_own_prompts_first(tmp_path, monkeypatch):
    monkeypatch.setattr(video_gen_cli, "PROMPTS_DIR", tmp_path)
    for name in ["a.txt", "prompt_s3_t2v.txt", "prompt_pih3_wuxia.txt", "prompt_pih3_t2v.txt", "prompt_pih3_bedroom.txt"]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    t2v = next(m for m in MODES if m.key == "t2v")
    assert list_prompt_choices(PiH3Profile(), t2v) == [
        "prompt_pih3_t2v.txt",
        "prompt_pih3_bedroom.txt",
        "prompt_pih3_wuxia.txt",
        "a.txt",
        "prompt_s3_t2v.txt",
    ]


@pytest.mark.parametrize("total, expected", [(10.0, [10.0]), (15.0, [15.0]), (20.0, [10.0, 10.0]), (40.0, [40 / 3] * 3)])
def test_split_into_segments_keeps_each_segment_within_limit(total, expected):
    assert split_into_segments(total) == pytest.approx(expected)
