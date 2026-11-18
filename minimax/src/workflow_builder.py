#!/usr/bin/env python3
"""
建立 MiniMax H3 HyperFlow Flash (6步生成，同步生成立體聲音訊與畫面) 與 SeedVR2 影片放大工作流 (ComfyUI API JSON 格式)。
"""

from typing import Optional


DEFAULT_UNET_NAME = "minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf"
DEFAULT_CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

DEFAULT_SEEDVR2_DIT = "seedvr2_ema_3b_fp16.safetensors"
DEFAULT_SEEDVR2_VAE = "ema_vae_fp16.safetensors"

CANVAS_MULTIPLE = 32


def seconds_to_h3_frames(seconds: float, fps: int = 24) -> int:
    """
    將秒數轉換為 MiniMax H3 所需的影格數 (4n + 4 或 4n 對齊，官方預設 5.2秒 = 124 frames)。
    """
    raw_frames = max(24, int(round(seconds * fps)))
    # 對齊到 4 的倍數 (例如 120 或 124 frames)
    aligned = ((raw_frames + 3) // 4) * 4
    return aligned


def nearest_4n_plus_1(n: int) -> int:
    """
    SeedVR2 batch_size 必須符合 4n + 1 規則 (1, 5, 9, 13, 17, 21, ...)。
    """
    if n <= 1:
        return 1
    k = max(1, (n - 1) // 4)
    return 4 * k + 1


def build_hyperflow_workflow(
    prompt: str,
    width: int = 1024,
    height: int = 768,
    length_frames: int = 124,
    steps: int = 6,
    seed: int = 42,
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
    建立 ComfyUI API 格式的 MiniMax H3 HyperFlow Flash (6步) 工作流：
    - 若 first_frame_image 為 None：執行 Text-to-Video + Audio (hyperflow_t2v)
    - 若提供 first_frame_image：執行 First-Frame Image-to-Video + Audio (hyperflow_i2v)
    - 若 upscale_with_seedvr2 為 True：在 VAEDecode 後串接 SeedVR2VideoUpscaler 將畫面放大，再與音軌合成影片
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
            "_meta": {"title": "HyperFlow Loader (14B Flash GGUF)"},
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
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": prompt,
                "width": gen_w,
                "height": gen_h,
                "length": length_frames,
            },
            "_meta": {"title": "MiniMax H3 Conditioning & Latent"},
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

    # 若為接續生成 (I2V)，加入 LoadImage 並連入 MiniMaxH3ImageToVideo.first_frame
    if first_frame_image:
        workflow["20"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": first_frame_image,
            },
            "_meta": {"title": "Load First Frame (Continuation)"},
        }
        workflow["5"]["inputs"]["first_frame"] = ["20", 0]

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
