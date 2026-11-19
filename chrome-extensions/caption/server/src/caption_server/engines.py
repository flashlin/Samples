import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from .audio_decoder import TARGET_SAMPLE_RATE


class SpeechRecognizer(Protocol):
    name: str

    def transcribe(self, samples: np.ndarray) -> str: ...


class Translator(Protocol):
    name: str

    def translate(self, text: str, keep_terms: list[str]) -> str: ...


class QwenAsrRecognizer:
    def __init__(self, model_id: str, language: str, device: str) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self.name = model_id
        self._language = language
        self._lock = threading.Lock()
        self._model = Qwen3ASRModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)

    def transcribe(self, samples: np.ndarray) -> str:
        with self._lock:
            result = self._model.transcribe(audio=(samples, TARGET_SAMPLE_RATE), language=self._language)
        return result[0].text.strip()


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
