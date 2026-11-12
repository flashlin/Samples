# coding=utf-8
# Copyright 2026 The NetEase Youdao team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""llama.cpp decoding for :class:`r2t2.R2T2ASRModel`, without modifying it.

``r2t2.R2T2ASRModel`` implements the streaming algorithm (audio buffering,
prefix rollback, punctuation normalisation, fixed-text extraction) against the
vLLM engine. All of that logic is backend-independent; the only thing that is
genuinely vLLM-specific is the single engine call it makes per chunk:

    outputs = self.model.generate([inp], sampling_params=..., use_tqdm=False)
    gen_text = outputs[0].outputs[0].text

So rather than forking or editing that algorithm, this module supplies a
``self.model`` that honours exactly that calling convention but decodes with
llama.cpp. The parent class is then reused verbatim -- streaming behaviour,
prompt construction, and the state object all stay in ``r2t2``.

Three decode routes are provided:

``LlamaHybrid``
    vLLM-native audio encoder + llama.cpp decoder. Best accuracy/latency
    trade-off, and the route covered by the smoke test.
``LlamaNative``
    In-process llama.cpp end to end (mel + encoder + decoder).

A third, ``llama-server``-backed route is available directly through
:class:`r2t2_llama.llama_native_backend.LlamaServerStreaming`. It is not
exposed here because ``llama-server`` takes chat messages rather than a
prebuilt prompt string, so it cannot consume the prompt this class forwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

# Upstream Qwen3-ASR config/processor come from the official `qwen-asr`
# package; the streaming model class comes from this repository's `r2t2`.
from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRProcessor
from r2t2 import R2T2ASRModel

from .debug import debug_print

# GGUF weights are not shipped with the package. By default both files are
# discovered inside ``gguf_dir``: the multimodal projector is the one whose name
# starts with ``mmproj``, the language model is the other ``*.gguf``. Pass
# explicit names to override.
MMPROJ_PREFIX = "mmproj"


# ---------------------------------------------------------------------------- #
# vLLM-shaped result objects
#
# The parent reads `outputs[0].outputs[0].text`. These two tiny holders let a
# llama.cpp result satisfy that access pattern.
# ---------------------------------------------------------------------------- #

class _Completion:
    __slots__ = ("text", "token_ids", "finish_reason")

    def __init__(self, text: str, token_ids=None, finish_reason=None):
        self.text = text
        self.token_ids = token_ids or []
        self.finish_reason = finish_reason


class _RequestOutput:
    __slots__ = ("outputs",)

    def __init__(self, completion: _Completion):
        self.outputs = [completion]


class LlamaEngineAdapter:
    """Presents a llama.cpp backend through vLLM's ``generate()`` signature.

    The parent passes the fully-built prompt (``state.prompt_raw + prefix``)
    and the accumulated audio inside the vLLM request dict. The llama.cpp
    backends want the prompt as a plain string plus the audio separately, so
    this adapter unpacks the request and forwards it.
    """

    def __init__(self, backend: Any, tokenizer: Any, max_new_tokens: int = 4096):
        self.backend = backend
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)

    # -- tokenizer passthrough (kept so the adapter can stand in for a engine
    #    whose tokenizer the caller may want to reach) --
    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def detokenize(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def generate(self, prompts, sampling_params=None, use_tqdm: bool = False):
        """vLLM-compatible entry point used by the parent's streaming loop."""
        if not prompts:
            raise ValueError("generate() requires exactly one request")
        request = prompts[0]

        prompt = request.get("prompt", "")
        audio_list = (request.get("multi_modal_data") or {}).get("audio") or []
        if not len(audio_list):
            raise ValueError("generate() requires audio in multi_modal_data")
        audio = audio_list[0]

        max_tokens = getattr(sampling_params, "max_tokens", None)
        max_tokens = int(max_tokens) if max_tokens else self.max_new_tokens

        text = self.backend.generate_from_prompt(
            audio=audio,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        debug_print(
            f"[llama] audio_samples={len(audio)} max_tokens={max_tokens} "
            f"text={text!r}"
        )
        return [_RequestOutput(_Completion(text))]


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #

def _load_processor(processor_path: str) -> Any:
    """Load the HF processor, aligning its window size with the model config.

    The llama.cpp routes deliberately keep tokenization on the HF processor so
    prompts and token boundaries match the vLLM route exactly.
    """
    processor = Qwen3ASRProcessor.from_pretrained(
        processor_path, fix_mistral_regex=True
    )
    config = Qwen3ASRConfig.from_pretrained(processor_path)
    processor.n_window = config.thinker_config.audio_config.n_window
    return processor


def _resolve_gguf(gguf_dir: str,
                  model_name: Optional[str] = None,
                  mmproj_name: Optional[str] = None):
    """Locate the language-model and projector GGUF files in ``gguf_dir``.

    Names are auto-discovered so the weights can be published under any
    filename; ``model_name`` / ``mmproj_name`` override the discovery.
    """
    root = Path(gguf_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"gguf directory not found: {root}")

    if model_name and mmproj_name:
        model_gguf = root / model_name
        mmproj_gguf = root / mmproj_name
    else:
        found = sorted(root.glob("*.gguf"))
        projectors = [p for p in found if p.name.startswith(MMPROJ_PREFIX)]
        models = [p for p in found if not p.name.startswith(MMPROJ_PREFIX)]
        if not projectors or not models:
            raise FileNotFoundError(
                f"expected one '{MMPROJ_PREFIX}*.gguf' projector and one other "
                f"'*.gguf' language model in {root}, found: "
                f"{[p.name for p in found] or 'no .gguf files'}"
            )
        model_gguf = root / (model_name or models[0].name)
        mmproj_gguf = root / (mmproj_name or projectors[0].name)

    if not model_gguf.exists():
        raise FileNotFoundError(f"model gguf not found: {model_gguf}")
    if not mmproj_gguf.exists():
        raise FileNotFoundError(f"projector gguf not found: {mmproj_gguf}")
    return str(model_gguf), str(mmproj_gguf)


# ---------------------------------------------------------------------------- #
# public model
# ---------------------------------------------------------------------------- #

class R2T2LlamaASRModel(R2T2ASRModel):
    """:class:`r2t2.R2T2ASRModel` streaming, decoded by llama.cpp.

    ``backend="vllm"`` is reported to the parent because its streaming methods
    assert on that value, and every call it makes under that assumption is
    served by :class:`LlamaEngineAdapter`. The attribute
    :attr:`decode_backend` records what actually performs decoding.
    """

    def __init__(
        self,
        model: LlamaEngineAdapter,
        processor: Any,
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 4096,
    ):
        super().__init__(
            backend="vllm",
            model=model,
            processor=processor,
            sampling_params=None,
            forced_aligner=None,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        self.decode_backend = "llama"

    # ----------------------------- constructors ----------------------------- #

    @classmethod
    def LlamaHybrid(
        cls,
        processor_path: str,
        gguf_dir: str,
        model_gguf_name: Optional[str] = None,
        mmproj_gguf_name: Optional[str] = None,
        max_new_tokens: int = 4096,
        **hybrid_kwargs: Any,
    ) -> "R2T2LlamaASRModel":
        """vLLM-native audio encoder + llama.cpp decoder."""
        from .llama_hybrid_backend import HybridConfig, LlamaHybridStreaming

        processor = _load_processor(processor_path)
        model_gguf, mmproj_gguf = _resolve_gguf(
            gguf_dir, model_gguf_name, mmproj_gguf_name
        )
        hybrid = LlamaHybridStreaming(
            HybridConfig(
                model=model_gguf,
                mmproj=mmproj_gguf,
                processor_path=processor_path,
                **hybrid_kwargs,
            ),
            tokenizer=processor.tokenizer,
        )
        return cls(
            model=LlamaEngineAdapter(hybrid, processor.tokenizer, max_new_tokens),
            processor=processor,
            max_new_tokens=max_new_tokens,
        )

    @classmethod
    def LlamaNative(
        cls,
        processor_path: str,
        gguf_dir: str,
        model_gguf_name: Optional[str] = None,
        mmproj_gguf_name: Optional[str] = None,
        n_ctx: int = 32768,
        n_batch: int = 8192,
        n_threads: int = 32,
        use_gpu: bool = True,
        n_gpu_layers: int = -1,
        max_new_tokens: int = 4096,
    ) -> "R2T2LlamaASRModel":
        """In-process llama.cpp streaming backend (no llama-server needed)."""
        from .llama_native_backend import LlamaNativeConfig, LlamaNativeStreaming

        processor = _load_processor(processor_path)
        model_gguf, mmproj_gguf = _resolve_gguf(
            gguf_dir, model_gguf_name, mmproj_gguf_name
        )
        cfg = LlamaNativeConfig(
            model=model_gguf,
            mmproj=mmproj_gguf,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            use_gpu=use_gpu,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_new_tokens,
        )
        native = LlamaNativeStreaming(cfg, tokenizer=processor.tokenizer)
        return cls(
            model=LlamaEngineAdapter(native, processor.tokenizer, max_new_tokens),
            processor=processor,
            max_new_tokens=max_new_tokens,
        )

    def transcribe(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "One-shot transcribe() is not routed through this class. Use "
            "LlamaNativeOnetime or LlamaServerClient from "
            "r2t2_llama.llama_native_backend for one-shot llama.cpp decoding."
        )


__all__ = ["R2T2LlamaASRModel", "LlamaEngineAdapter"]
