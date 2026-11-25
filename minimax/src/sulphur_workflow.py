from dataclasses import dataclass
from typing import Optional

from sulphur_assets import AUDIO_VAE_NAME, S3_CLIP_NAME, S3_CORRECTOR_NAME, S3_UNET_NAME, VIDEO_VAE_NAME

DEFAULT_STEPS = 20


@dataclass(frozen=True)
class SulphurJob:
    prompt: str
    width: int
    height: int
    length_frames: int
    seed: int
    filename_prefix: str
    steps: int = DEFAULT_STEPS
    use_corrector: bool = True
    first_frame_image: Optional[str] = None


def _node(class_type: str, title: str, **inputs) -> dict:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def _sampling_model_ref(job: SulphurJob) -> list:
    return ["18", 0] if job.use_corrector else ["2", 0]


def _shared_nodes(job: SulphurJob, guider_id: str, latent_ref: list) -> dict:
    nodes = {
        "1": _node("UNETLoader", "Sulphur 3 (W4A8)", unet_name=S3_UNET_NAME, weight_dtype="default"),
        "2": _node("MiniMaxH3SigmaShift", "Sigma shift 12 / 3", model=["1", 0], shift_video=12.0, shift_audio=3.0),
        "3": _node("CLIPLoader", "Text encoder (Qwen3-VL W4A8)", clip_name=S3_CLIP_NAME, type="minimax", device="default"),
        "4": _node("VAELoader", "Video VAE", vae_name=VIDEO_VAE_NAME),
        "5": _node("VAELoader", "Audio VAE", vae_name=AUDIO_VAE_NAME),
        "9": _node("BasicScheduler", f"Scheduler ({job.steps} steps)", model=["2", 0], scheduler="simple", steps=job.steps, denoise=1.0),
        "10": _node("KSamplerSelect", "Sampler (euler)", sampler_name="euler"),
        "11": _node("RandomNoise", "Noise seed", noise_seed=job.seed),
        "12": _node("SamplerCustomAdvanced", "Sampling", noise=["11", 0], guider=[guider_id, 0], sampler=["10", 0], sigmas=["9", 0], latent_image=latent_ref),
        "13": _node("VAEDecode", "Decode video", samples=["12", 0], vae=["4", 0]),
        "14": _node("VAEDecodeAudio", "Decode audio", samples=["12", 0], vae=["5", 0]),
        "15": _node("CreateVideo", "Combine video + audio", images=["13", 0], audio=["14", 0], fps=24.0),
        "16": _node("SaveVideo", "Save MP4", video=["15", 0], filename_prefix=job.filename_prefix, format="mp4", codec="auto"),
    }
    if job.use_corrector:
        nodes["17"] = _node("LoadH3Corrector", "Load H3 corrector", corrector=S3_CORRECTOR_NAME)
        nodes["18"] = _node(
            "ApplyH3Corrector", "Apply H3 corrector",
            model=["2", 0], corrector=["17", 0], strength=1.0, budget_fraction=0.5, velocity_floor=0.001, start_step=0, end_step=-1,
        )
    return nodes


def build_text_to_video_workflow(job: SulphurJob) -> dict:
    workflow = _shared_nodes(job, guider_id="19", latent_ref=["8", 0])
    workflow.update({
        "6": _node("CLIPTextEncode", "Positive prompt", clip=["3", 0], text=job.prompt),
        "7": _node("CLIPTextEncode", "Negative prompt (ignored at CFG 1)", clip=["3", 0], text=""),
        "8": _node("EmptyMiniMaxH3LatentAV", "Empty AV latent", width=job.width, height=job.height, length=job.length_frames),
        "19": _node("CFGGuider", "CFG guider (cfg 1)", model=_sampling_model_ref(job), positive=["6", 0], negative=["7", 0], cfg=1.0),
    })
    return workflow


def build_image_to_video_workflow(job: SulphurJob) -> dict:
    if not job.first_frame_image:
        raise ValueError("Image-to-video requires a first frame image")
    workflow = _shared_nodes(job, guider_id="23", latent_ref=["21", 1])
    workflow.update({
        "22": _node("LoadImage", "First frame", image=job.first_frame_image),
        "21": _node(
            "MiniMaxH3ImageToVideo", "Image to video",
            clip=["3", 0], vae=["4", 0], first_frame=["22", 0],
            prompt=job.prompt, width=job.width, height=job.height, length=job.length_frames,
        ),
        "23": _node("BasicGuider", "Basic guider", model=_sampling_model_ref(job), conditioning=["21", 0]),
    })
    return workflow
