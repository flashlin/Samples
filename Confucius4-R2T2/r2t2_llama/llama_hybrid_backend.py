"""混合后端：vLLM 的音频 encoder + llama.cpp 的 decode-only LLM。

用途：把"音频编码"和"文本解码"两段拆开归因。已知 case47 中 llama 全栈会漏识别
开头低音量的 "Bridge number two"，而同一段音频喂 vLLM 全栈不漏；单条实验显示只要
把 vLLM 算出的 audio embedding 灌进 llama 的 decode，漏词即消失。本模块把这个
单条实验推广成可跑批量 WER 的完整后端，用于量化"encoder 差异"到底吃掉多少 WER。

组成：
  - 前端 encoder：qwen_asr.core.vllm_backend.qwen3_asr.Qwen3ASRAudioEncoder
    （官方包的 vLLM-native 实现，单独实例化，不起 vLLM engine）
  - 后端 decode  ：native pybind 的 Qwen3ASRNative.generate_once_with_embedding
    （已验证与原生 generate_once 逐 token bit-exact）

对上层暴露 tokenize / detokenize / generate 三个方法，直接适配现有流式分发层
R2T2StreamingMixin，无需改 ext/streaming_mixin.py。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .llama_native_backend import build_asr_prompt


@dataclass
class HybridConfig:
    """混合后端配置。

    model/mmproj 指 GGUF（decode 侧用）；processor_path 指 HF 目录
    （encoder 权重 + mel feature_extractor + tokenizer 都从这里来）。
    """

    model: str
    mmproj: str
    processor_path: str
    n_ctx: int = 32768
    n_batch: int = 8192
    n_threads: int = 32
    use_gpu: bool = True
    n_gpu_layers: int = -1
    max_tokens: int = 4096
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


class VllmAudioEncoder:
    """单独实例化 vLLM-native 音频 encoder，不启动 vLLM engine。

    vLLM 的层依赖分布式并行组 + 一个 current vllm config 上下文，这里用
    world_size=1 的 gloo 组和默认 VllmConfig 满足，纯本地单卡前向。
    """

    def __init__(self, processor_path: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 master_port: Optional[str] = None):
        # 端口可通过环境变量 HYBRID_MASTER_PORT 覆盖：多进程并行跑不同数据集时
        # 必须各用不同端口，否则 gloo 组会 EADDRINUSE 起不来。
        port = master_port or os.environ.get("HYBRID_MASTER_PORT", "29577")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(port)

        from vllm.distributed import parallel_state
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.init_distributed_environment(
                world_size=1, rank=0, local_rank=0,
                distributed_init_method="env://", backend="gloo",
            )
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            )

        from vllm.config import VllmConfig, set_current_vllm_config
        from qwen_asr.core.vllm_backend.qwen3_asr import Qwen3ASRAudioEncoder
        from qwen_asr.core.transformers_backend import (
            Qwen3ASRConfig,
            Qwen3ASRProcessor,
        )
        from safetensors.torch import safe_open

        hf_config = Qwen3ASRConfig.from_pretrained(processor_path)
        audio_config = hf_config.thinker_config.audio_config

        self.processor = Qwen3ASRProcessor.from_pretrained(
            processor_path, fix_mistral_regex=True)
        self.processor.n_window = audio_config.n_window
        self.feature_extractor = self.processor.feature_extractor

        with set_current_vllm_config(VllmConfig()):
            encoder = Qwen3ASRAudioEncoder(audio_config)
        encoder = encoder.to(device=device, dtype=dtype)
        encoder.eval()

        weights = []
        st_path = os.path.join(processor_path, "model.safetensors")
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                for prefix in ("thinker.audio_tower.", "audio_tower."):
                    if name.startswith(prefix):
                        weights.append((name[len(prefix):], f.get_tensor(name)))
                        break
        if not weights:
            raise RuntimeError(f"未在 {st_path} 找到 audio_tower.* 权重")
        encoder.load_weights(iter(weights))

        self.encoder = encoder
        self.device = device
        self.dtype = dtype
        self.n_window = int(audio_config.n_window)

    @torch.no_grad()
    def encode(self, wav16k: np.ndarray) -> np.ndarray:
        """波形 -> audio embedding，形状 [n_tokens, n_embd]，float32。

        mel 提取与官方 processor 同口径：padding=True + truncation=False，
        保证长音频不被截到 30s。
        """
        wav = np.asarray(wav16k, dtype=np.float32).reshape(-1)
        feats = self.feature_extractor(
            wav,
            sampling_rate=16000,
            return_attention_mask=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_features = feats["input_features"].to(device=self.device, dtype=self.dtype)
        attn = feats["attention_mask"].to(self.device)
        feature_lens = attn.sum(dim=1)

        # 与 HF get_audio_features 同口径：逐条、按真实长度截断后送 encoder
        # （encoder 不做 batch 以保精度）
        n = int(feature_lens[0].item())
        feat_2d = input_features[0][:, :n]
        lens_1 = feature_lens[0:1]

        # aftercnn_lens 必须是"逐 chunk 求和"，不是整体套 3 层 CNN 公式
        chunk_size = self.n_window * 2
        chunk_lengths = [chunk_size] * (n // chunk_size)
        if n % chunk_size:
            chunk_lengths.append(n % chunk_size)
        chunk_t = torch.tensor(chunk_lengths, dtype=torch.long, device=self.device)
        aftercnn_lens = self.encoder._get_cnn_output_lengths(chunk_t).sum().unsqueeze(0)

        out = self.encoder.forward(feat_2d, lens_1, aftercnn_lens)
        if isinstance(out, tuple):
            out = out[0]
        embd = out.float().cpu().numpy()
        if embd.ndim == 3:
            embd = embd[0]
        return np.ascontiguousarray(embd, dtype=np.float32)


class LlamaHybridStreaming:
    """vLLM encoder + llama decode 的流式后端，接口对齐 LlamaNativeStreaming。"""

    def __init__(self, cfg: HybridConfig, tokenizer=None):
        for path, name in ((cfg.model, "model"), (cfg.mmproj, "mmproj")):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} 不存在: {path}")

        self.cfg = cfg
        self.audio_encoder = VllmAudioEncoder(
            cfg.processor_path, device=cfg.device, dtype=cfg.dtype)

        # tokenizer 用于流式回退，与 vLLM 侧同口径；未显式传入则复用 encoder 侧 processor
        self._tokenizer = tokenizer or self.audio_encoder.processor.tokenizer

        from .native.qwen3asr_native import Qwen3ASRNative
        # mmproj 仍需加载：decode 侧要用 mtmd context 判断 mrope / non-causal
        self._native = Qwen3ASRNative(
            cfg.model, cfg.mmproj, cfg.n_ctx, cfg.n_batch,
            cfg.n_threads, cfg.use_gpu, cfg.n_gpu_layers,
        )

    def tokenize(self, text: str) -> list:
        return self._tokenizer.encode(text)

    def detokenize(self, tokens: list) -> str:
        return self._tokenizer.decode(tokens)

    def generate(
        self,
        audio: np.ndarray,
        context: str = "",
        force_language: Optional[str] = None,
        assistant_prefix: str = "",
        max_tokens: Optional[int] = None,
    ) -> str:
        prompt = build_asr_prompt(context or "", force_language, assistant_prefix)
        return self.generate_from_prompt(
            audio=audio, prompt=prompt, max_tokens=max_tokens
        )

    def generate_from_prompt(
        self,
        audio: np.ndarray,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """解码一个已经构造好的 prompt。

        调用方（r2t2 的流式算法）已经用 HF chat template 拼好了 prompt
        （``state.prompt_raw + prefix``），这里不再自行拼接，避免两套
        prompt 构造口径不一致。
        """
        embd = self.audio_encoder.encode(audio)
        result = self._native.generate_once_with_embedding(
            embd, prompt,
            int(max_tokens) if max_tokens is not None else self.cfg.max_tokens,
        )
        return result["text"]
