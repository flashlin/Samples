#!/usr/bin/env python3
import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from apply_mac_optimizations import patch_h3_corrector_mps
from download_models import REQUIRED_MODELS, download_single_model

ROOT_DIR = Path(__file__).resolve().parent.parent
UI_WORKFLOWS_DIR = ROOT_DIR / "workflows"
SULPHUR_REPO = "SulphurAI/Sulphur-3"
CORRECTOR_NODE_DIR_NAME = "ComfyUI-H3-Corrector"

S3_UNET_NAME = "s3_w4a8.safetensors"
S3_CLIP_NAME = "qwen3vl_32b_minimax_h3_w4a8_int8down.safetensors"
S3_CORRECTOR_NAME = "s3_corrector.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"

SULPHUR_MODELS = [
    {
        "name": "Sulphur 3 diffusion model (W4A8)",
        "repo_id": SULPHUR_REPO,
        "repo_filename": f"quantized/{S3_UNET_NAME}",
        "target_subdir": "models/diffusion_models",
        "local_filename": S3_UNET_NAME,
    },
    {
        "name": "Qwen3-VL 32B MiniMax H3 text encoder (W4A8 + INT8 down_proj)",
        "repo_id": "JoaoZaokk/10Eros-Max-beta6-W4A8-ConvRot",
        "repo_filename": f"text_encoders/{S3_CLIP_NAME}",
        "target_subdir": "models/text_encoders",
        "local_filename": S3_CLIP_NAME,
    },
    {
        "name": "Sulphur 3 H3 corrector",
        "repo_id": SULPHUR_REPO,
        "repo_filename": S3_CORRECTOR_NAME,
        "target_subdir": "models/h3_correctors",
        "local_filename": S3_CORRECTOR_NAME,
    },
    *[cfg for cfg in REQUIRED_MODELS if cfg["local_filename"] in (VIDEO_VAE_NAME, AUDIO_VAE_NAME)],
]


def install_corrector_nodes(target_dir: Path) -> None:
    print(f"⬇️  Installing custom node {CORRECTOR_NODE_DIR_NAME} from {SULPHUR_REPO}/S3_ComfyUI.zip")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = hf_hub_download(repo_id=SULPHUR_REPO, filename="S3_ComfyUI.zip", local_dir=tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        shutil.copytree(Path(tmp) / CORRECTOR_NODE_DIR_NAME, target_dir)
    print(f"✅ Installed custom node: {target_dir}")


def ensure_corrector_nodes(comfy_dir: Path) -> None:
    target_dir = comfy_dir / "custom_nodes" / CORRECTOR_NODE_DIR_NAME
    if (target_dir / "__init__.py").is_file():
        print(f"✅ Custom node present: {CORRECTOR_NODE_DIR_NAME}")
    else:
        install_corrector_nodes(target_dir)
    patch_h3_corrector_mps(comfy_dir)


def ensure_ui_workflows(comfy_dir: Path) -> None:
    target_dir = comfy_dir / "user" / "default" / "workflows"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(UI_WORKFLOWS_DIR.glob("*.json")):
        shutil.copy2(source, target_dir / source.name)
        print(f"✅ ComfyUI workflow: {target_dir / source.name}")


def ensure_sulphur_assets(comfy_dir: Path, include_models: bool = True) -> bool:
    ensure_corrector_nodes(comfy_dir)
    ensure_ui_workflows(comfy_dir)
    if not include_models:
        return True
    return all([download_single_model(comfy_dir, cfg) for cfg in SULPHUR_MODELS])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Sulphur 3 models, the H3 corrector custom node and the ComfyUI workflows")
    parser.add_argument("--comfy-dir", default=str(ROOT_DIR / "ComfyUI"))
    parser.add_argument("--skip-models", action="store_true", help="Install the custom node and workflows only")
    args = parser.parse_args()
    if not ensure_sulphur_assets(Path(args.comfy_dir).resolve(), include_models=not args.skip_models):
        sys.exit(1)


if __name__ == "__main__":
    main()
