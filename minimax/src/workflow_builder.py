#!/usr/bin/env python3
"""
建立 MiniMax H3 Ref2VA HyperFlow Flash (6步生成，支援多張定裝參考圖與立體聲音訊) 與 SeedVR2 影片放大工作流 (ComfyUI API JSON 格式)。
"""

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


DEFAULT_UNET_NAME = "minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf"
DEFAULT_CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

DEFAULT_SEEDVR2_DIT = "seedvr2_ema_3b_fp16.safetensors"
DEFAULT_SEEDVR2_VAE = "ema_vae_fp16.safetensors"

CANVAS_MULTIPLE = 32
DEFAULT_REF_IMAGE_FILES = ("person1.jpg", "person2.jpg")
MAX_REF_IMAGES = 9

# 匹配提示詞中的參考圖檔名標籤，例如 <person1.jpg>、<person2.jpg>、<hero.png>
# 不會誤匹配 MiniMax 原生標籤 <Picture 1> / <Video 1> / <Audio 1>
REF_TAG_PATTERN = re.compile(
    r"<([^<>\r\n]+?\.(?:jpg|jpeg|png|webp|bmp))>",
    re.IGNORECASE,
)


def api_node(class_type: str, title: str, **inputs) -> dict:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def seconds_to_h3_frames(seconds: float, fps: int = 24) -> int:
    """
    將秒數轉換為 MiniMax H3 原生 17k + 5 影格網格 (24 fps；官方預設 5.2 秒 = 124 frames)。
    """
    raw_frames = max(22, int(round(seconds * fps)))
    while raw_frames % 17 != 5:
        raw_frames += 1
    return raw_frames


def nearest_4n_plus_1(n: int) -> int:
    """
    SeedVR2 batch_size 必須符合 4n + 1 規則 (1, 5, 9, 13, 17, 21, ...)。
    """
    if n <= 1:
        return 1
    k = max(1, (n - 1) // 4)
    return 4 * k + 1


def find_reference_image(filename: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    """
    依序在指定目錄中尋找參考圖片檔案，找不到時回傳 None。
    """
    clean_name = filename.strip()
    if not clean_name:
        return None

    candidate_path = Path(clean_name)
    if candidate_path.is_absolute():
        return candidate_path.resolve() if candidate_path.is_file() else None

    for base_dir in search_dirs:
        p = base_dir / clean_name
        if p.is_file():
            return p.resolve()
    return None


def parse_and_resolve_references(
    prompt_text: str,
    search_dirs: Sequence[Path],
    fixed_files: Sequence[str] = DEFAULT_REF_IMAGE_FILES,
    max_refs: int = MAX_REF_IMAGES,
) -> Tuple[str, List[Tuple[str, Path]], List[str]]:
    """
    解析固定參考圖 (person1.jpg, person2.jpg) 與提示詞中的 <檔名.jpg> 標籤：
    1. 固定優先檢查 person1.jpg、person2.jpg；若檔案存在即自動載入，不存在則忽略跳過。
    2. 掃描提示詞中所有 <xxx.jpg/png/webp/bmp> 標籤：
       - 若對應檔案存在，自動對應為 <Picture N>（不重複載入同一檔案），並將提示詞標籤替換為 <Picture N>。
       - 若對應檔案不存在，忽略跳過並移除該標籤，不中斷流程。
    回傳: (轉換後的提示詞, [(顯示名稱, 檔案絕對路徑), ...], [找不到的標籤清單])
    """
    resolved_refs: List[Tuple[str, Path]] = []
    path_to_pic_idx: dict[Path, int] = {}
    missing_tags: List[str] = []

    def register_image(label: str, img_path: Path) -> Optional[int]:
        if img_path in path_to_pic_idx:
            return path_to_pic_idx[img_path]
        if len(resolved_refs) >= max_refs:
            return None
        resolved_refs.append((label, img_path))
        pic_idx = len(resolved_refs)
        path_to_pic_idx[img_path] = pic_idx
        return pic_idx

    # 1. 固定檢查 person1.jpg、person2.jpg
    for fixed_name in fixed_files:
        found = find_reference_image(fixed_name, search_dirs)
        if found is not None:
            register_image(fixed_name, found)

    # 2. 解析提示詞中的 <filename.ext> 標籤並轉換為 <Picture N>
    def _replace_tag(match: re.Match) -> str:
        raw_name = match.group(1).strip()
        found = find_reference_image(raw_name, search_dirs)
        if found is None:
            if raw_name not in missing_tags:
                missing_tags.append(raw_name)
            return ""
        pic_idx = register_image(raw_name, found)
        if pic_idx is None:
            if raw_name not in missing_tags:
                missing_tags.append(f"{raw_name} (超過上限 {max_refs} 張)")
            return ""
        return f"<Picture {pic_idx}>"

    replaced_prompt = REF_TAG_PATTERN.sub(_replace_tag, prompt_text)

    # 清理移除不存在標籤後可能產生的多餘空白，但保留原本段落換行
    cleaned_lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in replaced_prompt.splitlines()]
    resolved_prompt = "\n".join(cleaned_lines).strip()

    return resolved_prompt, resolved_refs, missing_tags


def build_hyperflow_workflow(
    prompt: str,
    width: int = 1024,
    height: int = 768,
    length_frames: int = 124,
    steps: int = 6,
    seed: int = 42,
    ref_images: Optional[List[str]] = None,
    ref_image_size: str = "match",
    first_frame_image: Optional[str] = None,
    upscale_with_seedvr2: bool = False,
    target_short_edge: int = 1080,
    target_max_edge: int = 0,
    seedvr2_device: str = "mps",
    filename_prefix: str = "hyperflow/segment",
    unet_name: str = DEFAULT_UNET_NAME,
    clip_name: str = DEFAULT_CLIP_NAME,
    video_vae_name: str = DEFAULT_VIDEO_VAE,
    audio_vae_name: str = DEFAULT_AUDIO_VAE,
    seedvr2_dit_name: str = DEFAULT_SEEDVR2_DIT,
    seedvr2_vae_name: str = DEFAULT_SEEDVR2_VAE,
) -> dict:
    """
    建立 ComfyUI API 格式的 MiniMax H3 Ref2VA HyperFlow Flash (6步) 工作流：
    - 使用 MiniMaxH3ReferenceToVideo 載入人物定裝參考圖 (person1.jpg / person2.jpg / 提示詞 <檔名> 標籤)
    - 若 ref_images 為空：以純文字生片 (Ref2VA T2V) 模式執行
    - 若提供 first_frame_image（長影片第 2 段以上接續）：額外透過 MiniMaxH3AddGuide 在第 0 幀錨定上一段最後一格，同時保留人物參考圖
    - 若 upscale_with_seedvr2 為 True：在 VAEDecode 後串接 SeedVR2VideoUpscaler 放大，再與音軌合成影片
    """
    gen_w = max(256, (width // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    gen_h = max(256, (height // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

    workflow = {
        "1": {
            "class_type": "HyperFlowLoader",
            "inputs": {
                "unet_name": unet_name,
                "weights_in": "auto",
            },
            "_meta": {"title": "HyperFlow Loader (14B Ref2VA Flash GGUF)"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "minimax",
                "device": "default",
            },
            "_meta": {"title": "Text Encoder (Qwen3-VL MiniMax)"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": video_vae_name,
            },
            "_meta": {"title": "Video VAE"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": audio_vae_name,
            },
            "_meta": {"title": "Audio VAE"},
        },
        "5": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "audio_vae": ["4", 0],
                "prompt": prompt,
                "width": gen_w,
                "height": gen_h,
                "length": length_frames,
                "ref_image_size": ref_image_size,
            },
            "_meta": {"title": "MiniMax H3 Ref2VA Conditioning & Latent"},
        },
        "6": {
            "class_type": "HyperFlowSigmas",
            "inputs": {
                "model": ["1", 0],
                "steps": steps,
            },
            "_meta": {"title": f"HyperFlow Sigmas ({steps} steps)"},
        },
        "7": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler",
            },
            "_meta": {"title": "Sampler (euler)"},
        },
        "8": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": int(seed),
            },
            "_meta": {"title": "Random Noise Seed"},
        },
        "9": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["1", 0],
                "conditioning": ["5", 0],
            },
            "_meta": {"title": "BasicGuider (CFG=1.0)"},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["9", 0],
                "sampler": ["7", 0],
                "sigmas": ["6", 0],
                "latent_image": ["5", 1],
            },
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["3", 0],
            },
            "_meta": {"title": "Decode Video Frames"},
        },
        "12": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["4", 0],
            },
            "_meta": {"title": "Decode Stereo Audio"},
        },
    }

    # 載入人物定裝參考圖 (最多 9 張)，依序接至 MiniMaxH3ReferenceToVideo.ref_images.ref_image_i
    for idx, ref_img_name in enumerate((ref_images or [])[:MAX_REF_IMAGES]):
        node_id = str(20 + idx)
        workflow[node_id] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": ref_img_name,
            },
            "_meta": {"title": f"Load Reference <Picture {idx + 1}> ({ref_img_name})"},
        }
        workflow["5"]["inputs"][f"ref_images.ref_image_{idx}"] = [node_id, 0]

    # 若為長影片接續段，透過 MiniMaxH3AddGuide 在第 0 幀錨定上一段最後一格，同時保留人物參考圖鎖定
    if first_frame_image:
        workflow["29"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": first_frame_image,
            },
            "_meta": {"title": "Load First Frame (Continuation)"},
        }
        workflow["15"] = {
            "class_type": "MiniMaxH3AddGuide",
            "inputs": {
                "positive": ["5", 0],
                "vae": ["3", 0],
                "latent": ["5", 1],
                "image": ["29", 0],
                "frame_idx": 0,
            },
            "_meta": {"title": "Anchor Continuation First Frame"},
        }
        workflow["9"]["inputs"]["conditioning"] = ["15", 0]

    # 決定進入 CreateVideo 的影像來源 (直接來自 11 或經過 SeedVR2 放大)
    video_images_source = ["11", 0]

    if upscale_with_seedvr2:
        batch_4n1 = nearest_4n_plus_1(min(21, length_frames))
        workflow["30"] = {
            "class_type": "SeedVR2LoadDiTModel",
            "inputs": {
                "model": seedvr2_dit_name,
                "device": seedvr2_device,
                "blocks_to_swap": 0,
                "swap_io_components": False,
                "offload_device": "none",
                "cache_model": False,
                "attention_mode": "sdpa",
            },
            "_meta": {"title": "SeedVR2 Load 3B DiT Model"},
        }
        workflow["31"] = {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {
                "model": seedvr2_vae_name,
                "device": seedvr2_device,
                "encode_tiled": True,
                "encode_tile_size": 1024,
                "encode_tile_overlap": 128,
                "decode_tiled": True,
                "decode_tile_size": 1024,
                "decode_tile_overlap": 128,
                "tile_debug": "false",
                "offload_device": "none",
                "cache_model": False,
            },
            "_meta": {"title": "SeedVR2 Load VAE Model"},
        }
        workflow["32"] = {
            "class_type": "SeedVR2VideoUpscaler",
            "inputs": {
                "image": ["11", 0],
                "dit": ["30", 0],
                "vae": ["31", 0],
                "seed": int(seed),
                "resolution": int(target_short_edge),
                "max_resolution": int(target_max_edge),
                "batch_size": batch_4n1,
                "uniform_batch_size": True,
                "temporal_overlap": 2,
                "prepend_frames": 0,
                "color_correction": "lab",
                "input_noise_scale": 0.0,
                "latent_noise_scale": 0.0,
                "offload_device": "none",
                "enable_debug": False,
            },
            "_meta": {"title": "SeedVR2 Video Upscaler"},
        }
        video_images_source = ["32", 0]

    workflow["13"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": video_images_source,
            "audio": ["12", 0],
            "fps": 24.0,
        },
        "_meta": {"title": "Combine Video + Audio (24 fps)"},
    }

    workflow["14"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["13", 0],
            "filename_prefix": filename_prefix,
            "format": "mp4",
            "codec": "auto",
        },
        "_meta": {"title": "Save MP4 Output"},
    }

    return workflow
