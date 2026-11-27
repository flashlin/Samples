#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from pi_h3_assets import PI_H3_STEPS, adapter_name, ensure_pi_h3_assets
from pi_h3_workflow import build_pi_h3_workflow
from run_pipeline import COMFY_DIR
from video_gen_cli import VideoJob, WorkflowMode, run_cli


class PiH3Profile:
    key = "pih3"
    output_tag = "pih3"
    title = "pi-H3"
    description = "Generate a video with pi-H3 (MiniMax H3 FL2VA + pi-Flow DWD adapter)"
    default_size = (672, 384)
    default_seconds = 15.0
    default_steps = 4
    step_choices = PI_H3_STEPS
    prompt_template_dir = Path(__file__).resolve().parent / "prompt_templates" / "pih3"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        return None

    def summary(self, args: argparse.Namespace) -> str:
        return f"adapter {adapter_name(args.steps)}"

    def build_workflow(self, mode: WorkflowMode, job: VideoJob, args: argparse.Namespace) -> dict:
        return build_pi_h3_workflow(job)

    def ensure_assets(self, args: argparse.Namespace) -> bool:
        return ensure_pi_h3_assets(COMFY_DIR, [args.steps])


if __name__ == "__main__":
    try:
        run_cli(PiH3Profile())
    except KeyboardInterrupt:
        sys.exit(130)
