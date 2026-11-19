import os

from .settings import Settings


def required_models(settings: Settings) -> list[str]:
    return list(dict.fromkeys([settings.asr_model, settings.aligner_model, settings.punctuation_model, settings.vad_model, settings.translate_model]))


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    for model in required_models(Settings()):
        print(f"Downloading {model} into the Hugging Face cache", flush=True)
        print(f"Ready: {model} -> {snapshot_download(model)}", flush=True)
