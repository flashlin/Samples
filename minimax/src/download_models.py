#!/usr/bin/env python3
"""
下載並驗證本專案所需的 MiniMax H3 HyperFlow Flash (6步生成) 與 SeedVR2 模型權重。
支援透過 comfy-cli model download 或 huggingface_hub 斷點續傳與進度條顯示。
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


REQUIRED_MODELS = [
    {
        "name": "MiniMax H3 Ref2VA HyperFlow Flash 14B (Q4_K_M GGUF - 6步參考圖生片版)",
        "repo_id": "VideoRebirthAI/hyperflow-h3-flash",
        "repo_filename": "ref2va_gguf/minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf",
        "target_subdir": "models/diffusion_models",
        "local_filename": "minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf",
        "essential": True,
    },
    {
        "name": "Qwen3-VL 32B MiniMax H3 Text Encoder (NVFP4 AWQ)",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "repo_filename": "split_files/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "fallback_filenames": [
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        ],
        "target_subdir": "models/text_encoders",
        "local_filename": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "essential": True,
    },
    {
        "name": "MiniMax H3 Video VAE (FP16)",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "repo_filename": "split_files/vae/minimax_h3_video_vae_fp16.safetensors",
        "fallback_filenames": [
            "minimax_h3_video_vae_fp16.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors",
        ],
        "target_subdir": "models/vae",
        "local_filename": "minimax_h3_video_vae_fp16.safetensors",
        "essential": True,
    },
    {
        "name": "MiniMax H3 Audio VAE (FP32 - 32kHz 立體聲)",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "repo_filename": "split_files/vae/minimax_h3_audio_vae_fp32.safetensors",
        "fallback_filenames": [
            "minimax_h3_audio_vae_fp32.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
        ],
        "target_subdir": "models/vae",
        "local_filename": "minimax_h3_audio_vae_fp32.safetensors",
        "essential": True,
    },
    {
        "name": "SeedVR2 3B DiT Upscaler Model (FP16)",
        "repo_id": "numz/SeedVR2_comfyUI",
        "repo_filename": "seedvr2_ema_3b_fp16.safetensors",
        "target_subdir": "models/SEEDVR2",
        "local_filename": "seedvr2_ema_3b_fp16.safetensors",
        "essential": False,
    },
    {
        "name": "SeedVR2 EMA VAE (FP16)",
        "repo_id": "numz/SeedVR2_comfyUI",
        "repo_filename": "ema_vae_fp16.safetensors",
        "target_subdir": "models/SEEDVR2",
        "local_filename": "ema_vae_fp16.safetensors",
        "essential": False,
    },
]


def download_single_model(comfy_dir: Path, model_cfg: dict) -> bool:
    target_dir = comfy_dir / model_cfg["target_subdir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / model_cfg["local_filename"]

    if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
        size_gb = target_path.stat().st_size / (1024 ** 3)
        print(f"✅ 已存在 [{model_cfg['name']}]: {target_path.name} ({size_gb:.2f} GB)")
        return True

    print(f"\n⬇️  正在下載 [{model_cfg['name']}]...")
    print(f"   來源: {model_cfg['repo_id']} -> {target_path}")

    candidate_files = [model_cfg["repo_filename"]] + model_cfg.get("fallback_filenames", [])
    last_err = None

    for repo_file in candidate_files:
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_cfg["repo_id"],
                filename=repo_file,
                local_dir=str(target_dir / ".hf_download_cache"),
            )
            shutil.move(str(downloaded_path), str(target_path))
            cache_tmp = target_dir / ".hf_download_cache"
            if cache_tmp.exists():
                shutil.rmtree(cache_tmp, ignore_errors=True)
            size_gb = target_path.stat().st_size / (1024 ** 3)
            print(f"✅ 下載完成: {target_path.name} ({size_gb:.2f} GB)")
            return True
        except Exception as e:
            last_err = e
            continue

    print(f"❌ 下載失敗 [{model_cfg['name']}]: {last_err}", file=sys.stderr)
    return False


def ensure_models(comfy_dir: Path, include_seedvr2: bool = True) -> bool:
    all_ok = True
    for cfg in REQUIRED_MODELS:
        if not include_seedvr2 and not cfg["essential"]:
            continue
        ok = download_single_model(comfy_dir, cfg)
        if not ok and cfg["essential"]:
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="下載 MiniMax H3 HyperFlow 與 SeedVR2 模型")
    parser.add_argument(
        "--comfy-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "ComfyUI"),
        help="ComfyUI 安裝路徑",
    )
    parser.add_argument(
        "--skip-seedvr2",
        action="store_true",
        help="僅下載 MiniMax H3 核心模型，暫時跳過 SeedVR2 模型",
    )
    args = parser.parse_args()

    comfy_dir = Path(args.comfy_dir).resolve()
    ok = ensure_models(comfy_dir, include_seedvr2=not args.skip_seedvr2)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
