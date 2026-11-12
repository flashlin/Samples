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
"""llama.cpp decode backends for Confucius4-R2T2.

This package is additive: it reuses :mod:`r2t2` (the streaming algorithm) and
the official ``qwen-asr`` package (the base modeling code) as-is, and only adds
llama.cpp decoding on top. Nothing in :mod:`r2t2` is modified or shadowed.

Typical use::

    from r2t2_llama import R2T2LlamaASRModel

    asr = R2T2LlamaASRModel.LlamaHybrid(
        processor_path="/path/to/Confucius4-R2T2",
        gguf_dir="/path/to/gguf",
    )
    state = asr.init_streaming_state(language="Chinese", chunk_size_sec=0.16)
    _, text = asr.streaming_transcribe(chunk, state, max_new_tokens)

Prebuilt CUDA artifacts ship in ``native/`` (the pybind extension) and ``bin/``
(the llama.cpp shared libraries). See the README in this directory for the
pinned llama.cpp version and rebuild instructions.
"""

from .model import R2T2LlamaASRModel

__all__ = ["R2T2LlamaASRModel"]
__version__ = "0.1.0"
