#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from download_models import REQUIRED_MODELS, download_single_model
from sulphur_assets import ensure_ui_workflows
from workflow_builder import DEFAULT_AUDIO_VAE, DEFAULT_CLIP_NAME

ROOT_DIR = Path(__file__).resolve().parent.parent
PIFLOW_NODE_REPO = "https://github.com/Lakonik/ComfyUI-piFlow.git"
PIFLOW_NODE_DIR_NAME = "ComfyUI-piFlow"
MINIMAX_H3_REPO = "Comfy-Org/MiniMax-H3"

PI_H3_UNET_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
PI_H3_VIDEO_VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
PI_H3_CLIP_NAME = DEFAULT_CLIP_NAME
PI_H3_AUDIO_VAE_NAME = DEFAULT_AUDIO_VAE
PI_H3_STEPS = (2, 4)

BASE_MODELS = [
    {
        "name": "MiniMax H3 FL2VA pruned (INT8 ConvRot)",
        "repo_id": MINIMAX_H3_REPO,
        "repo_filename": f"diffusion_models/{PI_H3_UNET_NAME}",
        "target_subdir": "models/diffusion_models",
        "local_filename": PI_H3_UNET_NAME,
    },
    {
        "name": "MiniMax H3 video VAE (INT8 ConvRot)",
        "repo_id": MINIMAX_H3_REPO,
        "repo_filename": f"vae/{PI_H3_VIDEO_VAE_NAME}",
        "target_subdir": "models/vae",
        "local_filename": PI_H3_VIDEO_VAE_NAME,
    },
    *[cfg for cfg in REQUIRED_MODELS if cfg["local_filename"] in (PI_H3_CLIP_NAME, PI_H3_AUDIO_VAE_NAME)],
]


def adapter_name(steps: int) -> str:
    return f"pi_h3_{steps}step_dwd_adapter_int8_convrot.safetensors"


def adapter_model(steps: int) -> dict:
    return {
        "name": f"pi-H3 {steps}-step DWD adapter (INT8 ConvRot)",
        "repo_id": "Lakonik/pi-H3",
        "repo_filename": adapter_name(steps),
        "target_subdir": "models/loras",
        "local_filename": adapter_name(steps),
    }


def ensure_piflow_node(comfy_dir: Path) -> None:
    target_dir = comfy_dir / "custom_nodes" / PIFLOW_NODE_DIR_NAME
    if (target_dir / "__init__.py").is_file():
        print(f"✅ Custom node present: {PIFLOW_NODE_DIR_NAME}")
        return
    print(f"⬇️  Installing custom node {PIFLOW_NODE_DIR_NAME} from {PIFLOW_NODE_REPO}")
    subprocess.run(["git", "clone", PIFLOW_NODE_REPO, str(target_dir)], check=True)


def ensure_pi_h3_assets(comfy_dir: Path, steps: Sequence[int], include_models: bool = True) -> bool:
    ensure_piflow_node(comfy_dir)
    ensure_ui_workflows(comfy_dir)
    if not include_models:
        return True
    models = BASE_MODELS + [adapter_model(s) for s in steps]
    return all([download_single_model(comfy_dir, cfg) for cfg in models])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pi-H3 models, the ComfyUI-piFlow custom node and the ComfyUI workflows")
    parser.add_argument("--comfy-dir", default=str(ROOT_DIR / "ComfyUI"))
    parser.add_argument("--steps", type=int, nargs="+", choices=PI_H3_STEPS, default=[4], help="Adapters to download")
    parser.add_argument("--skip-models", action="store_true", help="Install the custom node and workflows only")
    args = parser.parse_args()
    if not ensure_pi_h3_assets(Path(args.comfy_dir).resolve(), args.steps, include_models=not args.skip_models):
        sys.exit(1)


if __name__ == "__main__":
    main()
