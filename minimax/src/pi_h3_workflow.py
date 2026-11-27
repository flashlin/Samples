from pi_h3_assets import PI_H3_AUDIO_VAE_NAME, PI_H3_CLIP_NAME, PI_H3_UNET_NAME, PI_H3_VIDEO_VAE_NAME, adapter_name
from video_gen_cli import VideoJob
from workflow_builder import api_node

MAX_RAW_T_BY_STEPS = {2: 0.93, 4: 0.96}


def _model_nodes(job: VideoJob) -> dict:
    return {
        "1": api_node("UNETLoader", "MiniMax H3 FL2VA (INT8 ConvRot)", unet_name=PI_H3_UNET_NAME, weight_dtype="default"),
        "39": api_node("Load LakonLab Adapter", f"pi-H3 {job.steps}-step adapter", model=["1", 0], adapter_name=adapter_name(job.steps), adapter_strength=1.0),
        "19": api_node("MiniMaxH3SigmaShift", "Sigma shift 12 / 3", model=["39", 0], shift_video=12.0, shift_audio=3.0),
        "20": api_node(
            "BlockSparseAttention", "Sparse attention (dense fallback off CUDA)",
            model=["19", 0], selection="sol-attn", **{"selection.tau": 1.0}, start_percent=0.0, end_percent=1.0,
            dense_blocks="0, 1", min_tokens=12288, extra_tokens=0, sink_conditioning="exact_kv_and_rows", verbose=False,
        ),
        "3": api_node("CLIPLoader", "Text encoder (Qwen3-VL NVFP4)", clip_name=PI_H3_CLIP_NAME, type="minimax", device="default"),
        "4": api_node("VAELoader", "Video VAE (encodes key frames)", vae_name=PI_H3_VIDEO_VAE_NAME),
        "5": api_node("VAELoader", "Audio VAE", vae_name=PI_H3_AUDIO_VAE_NAME),
    }


def _first_frame_nodes(job: VideoJob) -> dict:
    return {
        "21": api_node("LoadImage", "First frame", image=job.first_frame_image),
        "23": api_node("ImageScale", "Scale first frame", image=["21", 0], upscale_method="lanczos", width=job.width, height=job.height, crop="center"),
    }


def _conditioning_node(job: VideoJob) -> dict:
    inputs = dict(clip=["3", 0], vae=["4", 0], prompt=job.prompt, width=job.width, height=job.height, length=job.length_frames)
    if job.first_frame_image:
        inputs["first_frame"] = ["23", 0]
    return {"6": api_node("MiniMaxH3ImageToVideo", "FL2VA conditioning", **inputs)}


def _sampling_nodes(job: VideoJob) -> dict:
    return {
        "7": api_node("KSamplerSelect", "Sampler (euler)", sampler_name="euler"),
        "8": api_node(
            "pi-Flow Scheduler", f"pi-Flow scheduler ({job.steps} steps)",
            model=["20", 0], steps=job.steps, min_raw_t=0.0, max_raw_t=MAX_RAW_T_BY_STEPS[job.steps], final_step_size_scale=0.6,
        ),
        "9": api_node("BasicGuider", "Basic guider", model=["20", 0], conditioning=["6", 0]),
        "10": api_node("RandomNoise", "Noise seed", noise_seed=job.seed),
        "11": api_node("SamplerCustomAdvanced", "Sampling", noise=["10", 0], guider=["9", 0], sampler=["7", 0], sigmas=["8", 0], latent_image=["6", 1]),
    }


def _output_nodes(job: VideoJob) -> dict:
    return {
        "12": api_node("DWD Decode", "DWD decode video", samples=["11", 0]),
        "13": api_node("VAEDecodeAudio", "Decode audio", samples=["11", 0], vae=["5", 0]),
        "14": api_node("CreateVideo", "Combine video + audio", images=["12", 0], audio=["13", 0], fps=24.0),
        "15": api_node("SaveVideo", "Save MP4", video=["14", 0], filename_prefix=job.filename_prefix, format="mp4", codec="auto"),
    }


def build_pi_h3_workflow(job: VideoJob) -> dict:
    workflow = {**_model_nodes(job), **_conditioning_node(job), **_sampling_nodes(job), **_output_nodes(job)}
    if job.first_frame_image:
        workflow.update(_first_frame_nodes(job))
    return workflow
