import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import TARGET_SAMPLE_RATE
from .sentence_segmenter import TimedToken


@dataclass(frozen=True)
class Recognition:
    text: str
    tokens: list[TimedToken]


class SpeechRecognizer(Protocol):
    name: str

    def recognize(self, samples: np.ndarray, with_timestamps: bool) -> Recognition: ...


class Punctuator(Protocol):
    name: str

    def punctuate(self, text: str) -> str: ...


class Translator(Protocol):
    name: str

    def translate(self, text: str, keep_terms: list[str]) -> str: ...


class QwenAsrRecognizer:
    def __init__(self, model_id: str, language: str, device: str, aligner_id: str | None) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self.name = model_id
        self._language = language
        self._lock = threading.Lock()
        self._can_align = aligner_id is not None
        aligner_kwargs = {"dtype": torch.bfloat16, "device_map": device}
        self._model = Qwen3ASRModel.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device, forced_aligner=aligner_id, forced_aligner_kwargs=aligner_kwargs if aligner_id else None
        )

    def recognize(self, samples: np.ndarray, with_timestamps: bool) -> Recognition:
        align = with_timestamps and self._can_align
        with self._lock:
            result = self._model.transcribe(audio=(samples, TARGET_SAMPLE_RATE), language=self._language, return_time_stamps=align)[0]
        tokens = [TimedToken(item.text, item.start_time, item.end_time) for item in result.time_stamps] if align and result.time_stamps else []
        return Recognition(text=result.text.strip(), tokens=tokens)


class SherpaPunctuator:
    def __init__(self, model_id: str) -> None:
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        self.name = model_id
        model_dir = Path(snapshot_download(model_id))
        model_file = next(model_dir.glob("*.onnx"))
        config = sherpa_onnx.OfflinePunctuationConfig(model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=str(model_file)))
        self._lock = threading.Lock()
        self._model = sherpa_onnx.OfflinePunctuation(config)

    def punctuate(self, text: str) -> str:
        if not text.strip():
            return text
        with self._lock:
            return self._model.add_punctuation(text)


def _load_mlx_model(model_id: str) -> tuple[object, object]:
    from mlx_lm import load

    return load(model_id)


def _generate_with_mlx(model: object, tokenizer: object, prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate

    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


class HyMtTranslator:
    _INSTRUCTION = "将以下文本翻译为繁体中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
    _GLOSSARY_HEADER = "参考下面的翻译：\n{pairs}\n\n"

    def __init__(self, model_id: str, max_tokens: int) -> None:
        self.name = model_id
        self._max_tokens = max_tokens
        self._mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._model, self._tokenizer = self._mlx_thread.submit(_load_mlx_model, model_id).result()

    def _build_prompt(self, text: str, keep_terms: list[str]) -> str:
        glossary = ""
        if keep_terms:
            pairs = "\n".join(f"{term} 翻译成 {term}" for term in keep_terms)
            glossary = self._GLOSSARY_HEADER.format(pairs=pairs)
        message = glossary + self._INSTRUCTION.format(text=text)
        return self._tokenizer.apply_chat_template([{"role": "user", "content": message}], add_generation_prompt=True, tokenize=False)

    def translate(self, text: str, keep_terms: list[str]) -> str:
        prompt = self._build_prompt(text, keep_terms)
        return self._mlx_thread.submit(_generate_with_mlx, self._model, self._tokenizer, prompt, self._max_tokens).result().strip()


@dataclass(frozen=True)
class VadTuning:
    threshold: float
    min_silence_seconds: float
    min_speech_seconds: float
    max_speech_seconds: float


class SileroVadFactory:
    _BUFFER_SECONDS = 120

    def __init__(self, model_id: str, tuning: VadTuning) -> None:
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        self.name = model_id
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(Path(snapshot_download(model_id)) / "silero_vad.onnx")
        config.silero_vad.threshold = tuning.threshold
        config.silero_vad.min_silence_duration = tuning.min_silence_seconds
        config.silero_vad.min_speech_duration = tuning.min_speech_seconds
        config.silero_vad.max_speech_duration = tuning.max_speech_seconds
        config.sample_rate = TARGET_SAMPLE_RATE
        self._config = config
        self._sherpa = sherpa_onnx

    def __call__(self) -> object:
        return self._sherpa.VoiceActivityDetector(self._config, buffer_size_in_seconds=self._BUFFER_SECONDS)
