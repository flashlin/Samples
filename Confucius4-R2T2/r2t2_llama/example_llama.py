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
"""llama.cpp decode demo for Confucius4-R2T2.

The repository's own ``example.py`` covers the vLLM routes. This script covers
the llama.cpp routes and deliberately reuses ``example.py``'s streaming driver
and audio helpers so both demos exercise the same chunking schedule.

Usage::

    # hybrid: vLLM audio encoder + llama.cpp decoder (recommended)
    GGUF_DIR=/path/to/gguf python -m r2t2_llama.example_llama \\
        --audio resources/test.wav \\
        --model_path /path/to/Confucius4-R2T2 \\
        --infer_mode stream_llama_hybrid

    # in-process llama.cpp, end to end
    GGUF_DIR=/path/to/gguf python -m r2t2_llama.example_llama \\
        --audio resources/test.wav \\
        --model_path /path/to/Confucius4-R2T2 \\
        --infer_mode stream_llama

    # one-shot in-process llama.cpp
    python -m r2t2_llama.example_llama \\
        --audio resources/test.wav \\
        --model_path /path/to/gguf \\
        --infer_mode onetime_llama
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Reuse the repository's streaming driver and audio helpers. `example.py` lives
# at the repository root, so make sure it is importable when this module is run
# from an installed package rather than a source checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from example import (  # noqa: E402  (path set up above)
    _resample_to_16k,
    read_wav_bytes_with_librosa,
    run_streaming,
)

from r2t2_llama import R2T2LlamaASRModel  # noqa: E402
from r2t2_llama.model import _resolve_gguf  # noqa: E402

STREAMING_MODES = ("stream_llama_hybrid", "stream_llama")
ONETIME_MODES = ("onetime_llama",)


def parse_args():
    p = argparse.ArgumentParser("Confucius4-R2T2 llama.cpp demo (single file)")
    p.add_argument("--audio", type=str, required=True,
                   help="Path to the input audio file.")
    p.add_argument("--model_path", type=str, required=True,
                   help="HF model/processor directory (streaming modes read the "
                        "encoder weights, feature extractor and tokenizer from "
                        "here). For --infer_mode onetime_llama pass the GGUF "
                        "directory instead, since no HF assets are needed.")
    p.add_argument("--infer_mode", type=str, default="stream_llama_hybrid",
                   choices=STREAMING_MODES + ONETIME_MODES)
    p.add_argument("--language", type=str, default="Chinese",
                   help="Language hint (e.g. Chinese, English). None if not set.")
    p.add_argument("--chunk_size_ms", type=int, default=160,
                   help="Chunk size in milliseconds (default: 160).")
    p.add_argument("--lookahead_ms", type=int, default=160,
                   help="Lookahead size in milliseconds (default: 160).")
    p.add_argument("--unfixed_token_num", type=int, default=1,
                   help="Number of unfixed tokens (default: 1).")
    p.add_argument("--context", type=str, default="",
                   help="Context / hotword hint.")
    return p.parse_args()


def _require_gguf_dir(mode: str) -> str:
    gguf_dir = os.getenv("GGUF_DIR")
    if not gguf_dir:
        raise SystemExit(
            f"--infer_mode {mode} requires the GGUF_DIR environment variable "
            f"(directory holding the model + mmproj GGUF weights)"
        )
    return gguf_dir


def main() -> None:
    args = parse_args()

    print(f"Loading audio from {args.audio} ...")
    wav, sr = read_wav_bytes_with_librosa(args.audio)
    wav16k = _resample_to_16k(wav, sr)

    chunk_size_sec = args.chunk_size_ms / 1000.0
    language = (
        args.language
        if args.language and args.language != "None"
        else None
    )

    if args.infer_mode in ONETIME_MODES:
        # One-shot decoding talks to the native backend directly; it does not
        # need the streaming model wrapper.
        from r2t2_llama.llama_native_backend import (
            LlamaNativeConfig,
            LlamaNativeOnetime,
        )

        try:
            model_gguf, mmproj_gguf = _resolve_gguf(args.model_path)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc))

        backend = LlamaNativeOnetime(
            LlamaNativeConfig(model=model_gguf, mmproj=mmproj_gguf)
        )
        native_result = backend.generate_once(
            wav16k, language=language or "Chinese"
        )
        result = native_result["text"].strip()
        if os.getenv("DEBUG_PRINT", "0").strip().lower() in {"1", "true", "yes", "on"}:
            print("llama_native_result_json="
                  + json.dumps(native_result, ensure_ascii=False))
        print("\n===== Final Result (Onetime llama)=====")
        print(f"final_result={result}")
        return

    gguf_dir = _require_gguf_dir(args.infer_mode)
    print(f"Loading model from {args.model_path} (gguf: {gguf_dir}) ...")

    if args.infer_mode == "stream_llama_hybrid":
        asr = R2T2LlamaASRModel.LlamaHybrid(
            processor_path=args.model_path,
            gguf_dir=gguf_dir,
        )
        label = "Streaming llama-hybrid"
    else:
        asr = R2T2LlamaASRModel.LlamaNative(
            processor_path=args.model_path,
            gguf_dir=gguf_dir,
        )
        label = "Streaming llama"

    # Same streaming driver the vLLM demo uses.
    result = run_streaming(
        asr,
        wav16k,
        step_ms=args.chunk_size_ms,
        chunk_size_sec=chunk_size_sec,
        unfixed_token_num=args.unfixed_token_num,
        lookahead_ms=args.lookahead_ms,
        language=language,
        context=args.context,
    )
    print(f"\n===== Final Result ({label})=====")
    print(f"final_result={result}")


if __name__ == "__main__":
    main()
