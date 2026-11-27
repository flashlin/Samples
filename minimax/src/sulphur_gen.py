#!/usr/bin/env python3
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from run_pipeline import COMFY_DIR
from sulphur_assets import ensure_sulphur_assets
from sulphur_workflow import (
    DEFAULT_STEPS,
    SulphurJob,
    build_image_to_video_workflow,
    build_text_to_video_workflow,
)
from video_gen_cli import VideoJob, WorkflowMode, run_cli

BUILDERS = {
    "t2v": build_text_to_video_workflow,
    "i2v": build_image_to_video_workflow,
}


class SulphurProfile:
    key = "s3"
    output_tag = "sulphur3"
    title = "Sulphur 3"
    description = "Generate a video with Sulphur 3 (MiniMax H3) + H3 corrector"
    default_size = (672, 384)
    default_seconds = 5.0
    default_steps = DEFAULT_STEPS
    step_choices = None
    prompt_template_dir = Path(__file__).resolve().parent / "prompt_templates"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--no-corrector", action="store_true", help="Bypass LoadH3Corrector / ApplyH3Corrector")

    def summary(self, args: argparse.Namespace) -> str:
        return f"corrector {'off' if args.no_corrector else 'on'}"

    def build_workflow(self, mode: WorkflowMode, job: VideoJob, args: argparse.Namespace) -> dict:
        return BUILDERS[mode.key](SulphurJob(**asdict(job), use_corrector=not args.no_corrector))

    def ensure_assets(self, args: argparse.Namespace) -> bool:
        return ensure_sulphur_assets(COMFY_DIR)


if __name__ == "__main__":
    try:
        run_cli(SulphurProfile())
    except KeyboardInterrupt:
        sys.exit(130)
