import argparse
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

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
from workflow_builder import CANVAS_MULTIPLE, seconds_to_h3_frames

PROMPTS_DIR = ROOT_DIR / "prompts"
IMAGES_DIR = ROOT_DIR / "images"
COMFY_INPUT_DIR = COMFY_DIR / "input"

SEGMENT_MAX_SECONDS = 15.0
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass(frozen=True)
class WorkflowMode:
    key: str
    label: str
    needs_image: bool


TEXT_TO_VIDEO = WorkflowMode("t2v", "文生影片", needs_image=False)
IMAGE_TO_VIDEO = WorkflowMode("i2v", "圖生影片", needs_image=True)
MODES = [TEXT_TO_VIDEO, IMAGE_TO_VIDEO]


@dataclass(frozen=True)
class VideoJob:
    prompt: str
    width: int
    height: int
    length_frames: int
    seed: int
    filename_prefix: str
    steps: int
    first_frame_image: Optional[str] = None


class ModelProfile(Protocol):
    key: str
    output_tag: str
    title: str
    description: str
    default_size: Tuple[int, int]
    default_seconds: float
    default_steps: int
    step_choices: Optional[Sequence[int]]
    prompt_template_dir: Path

    def add_arguments(self, parser: argparse.ArgumentParser) -> None: ...

    def summary(self, args: argparse.Namespace) -> str: ...

    def build_workflow(self, mode: WorkflowMode, job: VideoJob, args: argparse.Namespace) -> dict: ...

    def ensure_assets(self, args: argparse.Namespace) -> bool: ...


@dataclass(frozen=True)
class InputSelection:
    mode: WorkflowMode
    prompt_path: Path
    image_path: Optional[Path]


@dataclass(frozen=True)
class RunSettings:
    profile: ModelProfile
    mode: WorkflowMode
    prompt: str
    width: int
    height: int
    segment_seconds: List[float]
    seed: int
    steps: int
    timestamp: str
    args: argparse.Namespace


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


def prompt_prefix(profile: ModelProfile) -> str:
    return f"prompt_{profile.key}_"


def default_prompt_file(profile: ModelProfile, mode: WorkflowMode) -> Path:
    return PROMPTS_DIR / f"{prompt_prefix(profile)}{mode.key}.txt"


def ensure_default_prompt(profile: ModelProfile, mode: WorkflowMode) -> None:
    target = default_prompt_file(profile, mode)
    if target.is_file():
        return
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile.prompt_template_dir / f"{mode.key}.txt", target)
    print(f"📝 Created {target} from the official {profile.title} example")


def list_prompt_choices(profile: ModelProfile, mode: WorkflowMode) -> List[str]:
    default_name = default_prompt_file(profile, mode).name
    others = sorted(p.name for p in PROMPTS_DIR.glob("*.txt") if p.name != default_name)
    own = [name for name in others if name.startswith(prompt_prefix(profile))]
    rest = [name for name in others if not name.startswith(prompt_prefix(profile))]
    return [default_name] + own + rest


def select_prompt_file(profile: ModelProfile, mode: WorkflowMode, cli_prompt_file: Optional[str]) -> Path:
    if cli_prompt_file:
        return resolve_existing_file(cli_prompt_file)
    ensure_default_prompt(profile, mode)
    return PROMPTS_DIR / fzf_pick(list_prompt_choices(profile, mode), "Prompt file>")


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


def default_size_for(image_path: Optional[Path], default_size: Tuple[int, int]) -> Tuple[int, int]:
    if image_path is None:
        return default_size
    with Image.open(image_path) as img:
        aspect = img.width / img.height
    height = math.sqrt(default_size[0] * default_size[1] / aspect)
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


def resolve_seconds(cli_seconds: Optional[float], default_seconds: float, non_interactive: bool) -> float:
    seconds = cli_seconds if cli_seconds is not None else float(
        ask(f"Duration in seconds (over {SEGMENT_MAX_SECONDS:g}s is split into equal segments)", f"{default_seconds:g}", non_interactive)
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


def first_frame_name(settings: RunSettings, suffix: str) -> str:
    return f"_{settings.profile.key}_{settings.timestamp}_first_frame{suffix}"


def continuation_frame_name(settings: RunSettings, index: int) -> str:
    return f"_{settings.profile.key}_{settings.timestamp}_last_frame_{index:02d}.png"


def segment_mode(settings: RunSettings, index: int) -> WorkflowMode:
    return settings.mode if index == 1 else IMAGE_TO_VIDEO


def build_segment_workflow(settings: RunSettings, index: int, first_frame: Optional[str]) -> dict:
    job = VideoJob(
        prompt=settings.prompt,
        width=settings.width,
        height=settings.height,
        length_frames=seconds_to_h3_frames(settings.segment_seconds[index - 1]),
        seed=settings.seed + index - 1,
        steps=settings.steps,
        filename_prefix=f"{settings.profile.output_tag}/{settings.timestamp}_{settings.mode.key}_seg{index:02d}",
        first_frame_image=first_frame,
    )
    return settings.profile.build_workflow(segment_mode(settings, index), job, settings.args)


def save_workflow_json(settings: RunSettings, index: int, workflow: dict) -> Path:
    WORKFLOWS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = WORKFLOWS_TMP_DIR / f"{settings.timestamp}_{settings.profile.output_tag}_{settings.mode.key}_seg{index:02d}.json"
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


def final_output_path(settings: RunSettings) -> Path:
    total_seconds = sum(settings.segment_seconds)
    name = f"{settings.profile.output_tag}_{settings.mode.key}_{settings.width}x{settings.height}_{total_seconds:g}s_{settings.timestamp}.mp4"
    return OUTPUTS_DIR / name


def run_generation(settings: RunSettings, first_frame: Optional[str]) -> Path:
    if not settings.profile.ensure_assets(settings.args):
        raise SystemExit("Model download failed. See the messages above.")
    server_proc = start_comfy_server_via_cli()
    try:
        return concatenate_segments(generate_segments(settings, first_frame), final_output_path(settings))
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()


def parse_args(profile: ModelProfile) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=profile.description)
    parser.add_argument("--workflow", choices=[m.key for m in MODES], help="Skip fzf: t2v or i2v")
    parser.add_argument("--prompt-file", help="Skip fzf: prompt text file (default: pick from prompts/)")
    parser.add_argument("--image", help="Skip fzf: first frame for i2v (default: pick from images/)")
    parser.add_argument("--resolution", help=f"WxH, e.g. {profile.default_size[0]}x{profile.default_size[1]}")
    parser.add_argument("--duration", type=float, help="Total seconds; over 15s is split into equal segments")
    parser.add_argument("--steps", type=int, default=profile.default_steps, choices=profile.step_choices)
    parser.add_argument("--seed", type=int)
    profile.add_arguments(parser)
    parser.add_argument("--yes", "-y", action="store_true", help="Use defaults for resolution and duration")
    parser.add_argument("--dry-run", action="store_true", help="Write the API workflow JSON only")
    return parser.parse_args()


def describe_settings(settings: RunSettings) -> str:
    segments_text = " + ".join(f"{s:.1f}s" for s in settings.segment_seconds)
    details = [
        f"{settings.width}x{settings.height}",
        segments_text,
        f"{settings.steps} steps",
        f"seed {settings.seed}",
        settings.profile.summary(settings.args),
    ]
    return "📐 " + ", ".join(d for d in details if d)


def select_inputs(profile: ModelProfile, args: argparse.Namespace) -> InputSelection:
    mode = select_mode(args.workflow)
    prompt_path = select_prompt_file(profile, mode, args.prompt_file)
    image_path = select_first_frame(args.image) if mode.needs_image else None
    return InputSelection(mode=mode, prompt_path=prompt_path, image_path=image_path)


def build_settings(profile: ModelProfile, args: argparse.Namespace, inputs: InputSelection) -> RunSettings:
    width, height = resolve_size(args.resolution, default_size_for(inputs.image_path, profile.default_size), args.yes)
    return RunSettings(
        profile=profile,
        mode=inputs.mode,
        prompt=read_prompt_file(inputs.prompt_path),
        width=width,
        height=height,
        segment_seconds=split_into_segments(resolve_seconds(args.duration, profile.default_seconds, args.yes)),
        seed=args.seed if args.seed is not None else random.randint(1, 2**31 - 1),
        steps=args.steps,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        args=args,
    )


def run_cli(profile: ModelProfile) -> None:
    args = parse_args(profile)
    inputs = select_inputs(profile, args)

    print(f"🎬 {profile.title} - {inputs.mode.label} ({inputs.mode.key})")
    print(f"📄 Prompt file: {inputs.prompt_path}")
    if inputs.image_path:
        print(f"🖼️  First frame: {inputs.image_path}")

    settings = build_settings(profile, args, inputs)
    print(describe_settings(settings))

    first_frame = first_frame_name(settings, inputs.image_path.suffix.lower()) if inputs.image_path else None
    if args.dry_run:
        write_dry_run_workflows(settings, first_frame)
        return
    if inputs.image_path:
        stage_input_image(inputs.image_path, first_frame)
    final_path = run_generation(settings, first_frame)
    print(f"🎉 Done: {final_path} ({final_path.stat().st_size / 1024 / 1024:.1f} MB)")
