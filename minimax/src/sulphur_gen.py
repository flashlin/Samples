#!/usr/bin/env python3
import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image

from run_pipeline import (
    COMFY_DIR,
    OUTPUTS_DIR,
    ROOT_DIR,
    WORKFLOWS_TMP_DIR,
    concatenate_segments,
    execute_workflow_with_progress,
    extract_last_frame,
    parse_resolution,
    read_prompt_file,
    start_comfy_server_via_cli,
)
from sulphur_assets import ensure_sulphur_assets
from sulphur_workflow import (
    DEFAULT_STEPS,
    SulphurJob,
    build_image_to_video_workflow,
    build_text_to_video_workflow,
)
from workflow_builder import CANVAS_MULTIPLE, seconds_to_h3_frames

PROMPTS_DIR = ROOT_DIR / "prompts"
IMAGES_DIR = ROOT_DIR / "images"
PROMPT_TEMPLATES_DIR = Path(__file__).resolve().parent / "prompt_templates"
COMFY_INPUT_DIR = COMFY_DIR / "input"

DEFAULT_WIDTH = 672
DEFAULT_HEIGHT = 384
DEFAULT_SECONDS = 5.0
SEGMENT_MAX_SECONDS = 15.0
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass(frozen=True)
class WorkflowMode:
    key: str
    label: str
    build: Callable[[SulphurJob], dict]
    needs_image: bool

    @property
    def default_prompt_file(self) -> Path:
        return PROMPTS_DIR / f"prompt_s3_{self.key}.txt"

    @property
    def prompt_template(self) -> Path:
        return PROMPT_TEMPLATES_DIR / f"{self.key}.txt"


@dataclass(frozen=True)
class RunSettings:
    mode: WorkflowMode
    prompt: str
    width: int
    height: int
    segment_seconds: List[float]
    seed: int
    steps: int
    use_corrector: bool
    timestamp: str


MODES = [
    WorkflowMode("t2v", "文生影片", build_text_to_video_workflow, needs_image=False),
    WorkflowMode("i2v", "圖生影片", build_image_to_video_workflow, needs_image=True),
]


def fzf_pick(options: List[str], prompt: str) -> str:
    if shutil.which("fzf") is None:
        raise SystemExit("fzf not found. Install it with: brew install fzf")
    result = subprocess.run(
        ["fzf", f"--prompt={prompt} ", "--height=40%", "--reverse", "--no-multi"],
        input="\n".join(options), stdout=subprocess.PIPE, text=True,
    )
    choice = result.stdout.strip()
    if result.returncode != 0 or not choice:
        raise SystemExit("Cancelled.")
    return choice


def select_mode(cli_mode: Optional[str]) -> WorkflowMode:
    if cli_mode:
        return next(m for m in MODES if m.key == cli_mode)
    label = fzf_pick([m.label for m in MODES], "Workflow>")
    return next(m for m in MODES if m.label == label)


def resolve_existing_file(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    return path


def ensure_default_prompt(mode: WorkflowMode) -> None:
    if mode.default_prompt_file.is_file():
        return
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mode.prompt_template, mode.default_prompt_file)
    print(f"📝 Created {mode.default_prompt_file} from the official Sulphur 3 example")


def select_prompt_file(mode: WorkflowMode, cli_prompt_file: Optional[str]) -> Path:
    if cli_prompt_file:
        return resolve_existing_file(cli_prompt_file)
    ensure_default_prompt(mode)
    others = sorted(p for p in PROMPTS_DIR.glob("*.txt") if p != mode.default_prompt_file)
    names = [mode.default_prompt_file.name] + [p.name for p in others]
    return PROMPTS_DIR / fzf_pick(names, "Prompt file>")


def select_first_frame(cli_image: Optional[str]) -> Path:
    if cli_image:
        return resolve_existing_file(cli_image)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(p.name for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not candidates:
        raise SystemExit(f"No image in {IMAGES_DIR}. Put a .jpg/.png there or pass --image <path>.")
    return IMAGES_DIR / fzf_pick(candidates, "First frame>")


def snap_to_canvas(value: float) -> int:
    return max(CANVAS_MULTIPLE, int(round(value / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def default_size_for(image_path: Optional[Path]) -> Tuple[int, int]:
    if image_path is None:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT
    with Image.open(image_path) as img:
        aspect = img.width / img.height
    height = math.sqrt(DEFAULT_WIDTH * DEFAULT_HEIGHT / aspect)
    return snap_to_canvas(height * aspect), snap_to_canvas(height)


def ask(question: str, default: str, non_interactive: bool) -> str:
    if non_interactive:
        return default
    try:
        answer = input(f"👉 {question} [{default}]: ").strip()
    except EOFError:
        answer = ""
    return answer or default


def resolve_size(cli_resolution: Optional[str], default_size: Tuple[int, int], non_interactive: bool) -> Tuple[int, int]:
    default_text = f"{default_size[0]}x{default_size[1]}"
    raw = cli_resolution or ask("Resolution (WxH, multiples of 32)", default_text, non_interactive)
    width, height = parse_resolution(raw)
    return snap_to_canvas(width), snap_to_canvas(height)


def resolve_seconds(cli_seconds: Optional[float], non_interactive: bool) -> float:
    seconds = cli_seconds if cli_seconds is not None else float(
        ask(f"Duration in seconds (over {SEGMENT_MAX_SECONDS:g}s is split into equal segments)", f"{DEFAULT_SECONDS:g}", non_interactive)
    )
    if seconds <= 0:
        raise SystemExit("Duration must be greater than 0")
    return seconds


def split_into_segments(total_seconds: float) -> List[float]:
    count = math.ceil(total_seconds / SEGMENT_MAX_SECONDS)
    return [total_seconds / count] * count


def stage_input_image(image_path: Path, staged_name: str) -> str:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, COMFY_INPUT_DIR / staged_name)
    return staged_name


def continuation_frame_name(settings: RunSettings, index: int) -> str:
    return f"_s3_{settings.timestamp}_last_frame_{index:02d}.png"


def build_segment_workflow(settings: RunSettings, index: int, first_frame: Optional[str]) -> dict:
    job = SulphurJob(
        prompt=settings.prompt,
        width=settings.width,
        height=settings.height,
        length_frames=seconds_to_h3_frames(settings.segment_seconds[index - 1]),
        seed=settings.seed + index - 1,
        steps=settings.steps,
        use_corrector=settings.use_corrector,
        filename_prefix=f"sulphur3/{settings.timestamp}_{settings.mode.key}_seg{index:02d}",
        first_frame_image=first_frame,
    )
    build = settings.mode.build if index == 1 else build_image_to_video_workflow
    return build(job)


def save_workflow_json(settings: RunSettings, index: int, workflow: dict) -> Path:
    WORKFLOWS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = WORKFLOWS_TMP_DIR / f"{settings.timestamp}_sulphur3_{settings.mode.key}_seg{index:02d}.json"
    path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_dry_run_workflows(settings: RunSettings, first_frame: Optional[str]) -> None:
    for index in range(1, len(settings.segment_seconds) + 1):
        frame = first_frame if index == 1 else continuation_frame_name(settings, index - 1)
        print(f"🧾 Workflow JSON: {save_workflow_json(settings, index, build_segment_workflow(settings, index, frame))}")


def generate_segments(settings: RunSettings, first_frame: Optional[str]) -> List[Path]:
    videos: List[Path] = []
    total = len(settings.segment_seconds)
    for index in range(1, total + 1):
        print(f"\n▶️  Segment {index}/{total} ({settings.segment_seconds[index - 1]:.1f}s, first frame: {first_frame or 'none'})")
        workflow = build_segment_workflow(settings, index, first_frame)
        save_workflow_json(settings, index, workflow)
        video = execute_workflow_with_progress(workflow=workflow, segment_idx=index, total_segments=total)
        videos.append(video)
        if index < total:
            first_frame = continuation_frame_name(settings, index)
            extract_last_frame(video, COMFY_INPUT_DIR / first_frame)
    return videos


def run_generation(settings: RunSettings, first_frame: Optional[str]) -> Path:
    if not ensure_sulphur_assets(COMFY_DIR):
        raise SystemExit("Model download failed. See the messages above.")
    server_proc = start_comfy_server_via_cli()
    try:
        videos = generate_segments(settings, first_frame)
        total_seconds = sum(settings.segment_seconds)
        final_path = OUTPUTS_DIR / f"sulphur3_{settings.mode.key}_{settings.width}x{settings.height}_{total_seconds:g}s_{settings.timestamp}.mp4"
        return concatenate_segments(videos, final_path)
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with Sulphur 3 (MiniMax H3) + H3 corrector")
    parser.add_argument("--workflow", choices=[m.key for m in MODES], help="Skip fzf: t2v or i2v")
    parser.add_argument("--prompt-file", help="Skip fzf: prompt text file (default: pick from prompts/)")
    parser.add_argument("--image", help="Skip fzf: first frame for i2v (default: pick from images/)")
    parser.add_argument("--resolution", help="WxH, e.g. 672x384")
    parser.add_argument("--duration", type=float, help="Total seconds; over 15s is split into equal segments")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-corrector", action="store_true", help="Bypass LoadH3Corrector / ApplyH3Corrector")
    parser.add_argument("--yes", "-y", action="store_true", help="Use defaults for resolution and duration")
    parser.add_argument("--dry-run", action="store_true", help="Write the API workflow JSON only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = select_mode(args.workflow)
    prompt_path = select_prompt_file(mode, args.prompt_file)
    image_path = select_first_frame(args.image) if mode.needs_image else None

    print(f"🎬 Sulphur 3 - {mode.label} ({mode.key})")
    print(f"📄 Prompt file: {prompt_path}")
    if image_path:
        print(f"🖼️  First frame: {image_path}")

    width, height = resolve_size(args.resolution, default_size_for(image_path), args.yes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings = RunSettings(
        mode=mode,
        prompt=read_prompt_file(prompt_path),
        width=width,
        height=height,
        segment_seconds=split_into_segments(resolve_seconds(args.duration, args.yes)),
        seed=args.seed if args.seed is not None else random.randint(1, 2**31 - 1),
        steps=args.steps,
        use_corrector=not args.no_corrector,
        timestamp=timestamp,
    )
    segments_text = " + ".join(f"{s:.1f}s" for s in settings.segment_seconds)
    print(f"📐 {width}x{height}, {segments_text}, {settings.steps} steps, seed {settings.seed}, corrector {'on' if settings.use_corrector else 'off'}")

    first_frame_name = f"_s3_{timestamp}_first_frame{image_path.suffix.lower()}" if image_path else None
    if args.dry_run:
        write_dry_run_workflows(settings, first_frame_name)
        return
    if image_path:
        stage_input_image(image_path, first_frame_name)
    final_path = run_generation(settings, first_frame_name)
    print(f"🎉 Done: {final_path} ({final_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
