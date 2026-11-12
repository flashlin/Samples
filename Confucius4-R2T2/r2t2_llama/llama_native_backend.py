# coding=utf-8
"""In-process llama.cpp backends for Qwen3-ASR."""

from __future__ import annotations

import base64
import io
import os
import re

import numpy as np
import requests
import soundfile as sf

from dataclasses import dataclass
from typing import Optional

from qwen_asr.inference.utils import parse_asr_output

_ASR_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m.role + '\\n' }}"
    "{% if m.content is string %}{{ m.content }}"
    "{% else %}{% for c in m.content %}"
    "{% if c.type == 'text' %}{{ c.text }}"
    "{% elif c.type == 'input_audio' %}<|audio_start|><|audio_pad|><|audio_end|>"
    "{% endif %}{% endfor %}{% endif %}"
    "{% if m.role != 'assistant' %}<|im_end|>\\n{% endif %}" 
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

@dataclass
class LlamaNativeConfig:
    """进程内 pybind backend 配置；不启动 llama-server 或子进程。

    model / mmproj 为 GGUF 权重路径，必填（权重不随仓库分发）。
    """

    model: str
    mmproj: str
    n_ctx: int = 32768
    n_batch: int = 8192
    n_threads: int = 32
    use_gpu: bool = True
    n_gpu_layers: int = -1
    max_tokens: int = 4096


def build_asr_prompt(
    context: str = "",
    language: Optional[str] = "Chinese",
    assistant_prefix: str = "",
) -> str:
    """构造与 R2T2ASRModel._build_text_prompt 相同的 canonical prompt。"""
    prompt = (
        "<|im_start|>system\n"
        f"{context or ''}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if language:
        prompt += f"language {language}<asr_text>"
    return prompt + assistant_prefix


class LlamaNativeOnetime:
    """直接调用 pybind11 扩展中的 libllama/libmtmd。"""

    def __init__(self, cfg: LlamaNativeConfig):
        self.cfg = cfg
        for path, name in ((self.cfg.model, "model"), (self.cfg.mmproj, "mmproj")):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} 不存在: {path}")
        try:
            from .native.qwen3asr_native import Qwen3ASRNative
        except ImportError as exc:
            raise ImportError(
                "未找到 qwen3asr_native，请先构建 native/ 目录中的 pybind 扩展"
            ) from exc
        self._native = Qwen3ASRNative(
            self.cfg.model,
            self.cfg.mmproj,
            self.cfg.n_ctx,
            self.cfg.n_batch,
            self.cfg.n_threads,
            self.cfg.use_gpu,
            self.cfg.n_gpu_layers,
        )

    def generate_once(
        self,
        wav16k: np.ndarray,
        context: str = "",
        language: Optional[str] = "Chinese",
        assistant_prefix: str = "",
        max_tokens: Optional[int] = None,
    ) -> dict:
        prompt = build_asr_prompt(context, language, assistant_prefix)
        return self._generate_with_prompt(wav16k, prompt, max_tokens)

    def _generate_with_prompt(
        self,
        wav16k: np.ndarray,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> dict:
        audio = np.asarray(wav16k, dtype=np.float32).reshape(-1)
        return self._native.generate_once(
            audio,
            prompt,
            int(max_tokens or self.cfg.max_tokens),
        )

    def generate_from_prompt(
        self,
        audio: np.ndarray,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """解码一个已经构造好的 prompt，返回文本。

        调用方（r2t2 的流式算法）已用 HF chat template 拼好 prompt，
        这里不再自行拼接，保证与 vLLM 路线同口径。
        """
        return self._generate_with_prompt(audio, prompt, max_tokens)["text"]

    def transcribe(
        self,
        wav16k: np.ndarray,
        language: str = "Chinese",
    ) -> str:
        result = self.generate_once(wav16k, language=language)
        return result["text"].strip()


class LlamaNativeStreaming(LlamaNativeOnetime):
    """进程内 native 流式后端。

    为公共流式回退状态机补齐 tokenize/detokenize/generate 接口。
    实际生成仍由 native generate_once 完成，回退使用 HF tokenizer，
    与 vLLM 流式路径保持同一套 token rollback 规则。
    """

    def __init__(self, cfg: LlamaNativeConfig, tokenizer=None):
        super().__init__(cfg)
        if tokenizer is None:
            raise ValueError("LlamaNativeStreaming 需要 HF tokenizer 做回退")
        self._tokenizer = tokenizer

    def tokenize(self, text: str) -> list:
        # 如果用llama.cpp的tokenizer:
        # return list(
        #     self._native.tokenize(
        #         text,
        #         add_special=False,
        #         parse_special=True,
        #     )
        # )
        return self._tokenizer.encode(text)

    def detokenize(self, tokens: list) -> str:
        # 如果用llama.cpp的detokenizer
        # return self._native.detokenize(
        #     [int(token) for token in tokens],
        #     special=True,
        # )
        return self._tokenizer.decode(tokens)

    def generate(
        self,
        audio: np.ndarray,
        context: str = "",
        force_language: Optional[str] = None,
        assistant_prefix: str = "",
        max_tokens: Optional[int] = None,
    ) -> dict:
        # native generate_once 的输出天然是 prompt（含 prefix）之后的 delta。
        result = self.generate_once(
            audio,
            context=context or "",
            language=force_language,
            assistant_prefix=assistant_prefix,
            max_tokens=max_tokens,
        )
        return result


class LlamaServerClient:
    """
    常驻 llama-server 的 HTTP 客户端。
    接口与 LlamaMtmdOnetime.transcribe 保持一致，方便上层替换。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 28080,
                 timeout: float = 120.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self._health_check()

    def _health_check(self) -> None:
        try:
            r = requests.get(f"{self.base}/health", timeout=5)
        except requests.RequestException as e:
            raise RuntimeError(f"无法连接 llama-server @ {self.base}: {e}")
        if r.status_code != 200:
            raise RuntimeError(f"llama-server health 返回 {r.status_code}")

    @staticmethod
    def _wav_to_b64(wav16k: np.ndarray) -> str:
        if sf is None:
            raise ImportError("需要 soundfile: pip install soundfile")
        buf = io.BytesIO()
        sf.write(buf, wav16k, 16000, format="WAV")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def transcribe_raw(
        self,
        wav16k: np.ndarray,
        prompt: str = "Transcribe the audio.",
        max_tokens: int = 4096,
    ) -> str:
        """不剥 'language X<asr_text>' 前缀的原始输出，流式里要用。"""
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio",
                     "input_audio": {"data": self._wav_to_b64(wav16k), "format": "wav"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        r = requests.post(f"{self.base}/v1/chat/completions",
                          json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def transcribe(self, wav16k: np.ndarray, language: str = "Chinese") -> str:
        text = self.transcribe_raw(wav16k)
        return re.sub(r"^\s*language\s+\S+\s*<asr_text>\s*", "", text).strip()
    
    def tokenize(self, text: str) -> list:
        r = requests.post(f"{self.base}/tokenize",
                        json={"content": text}, timeout=30)
        r.raise_for_status()
        return r.json().get("tokens", [])

    def detokenize(self, tokens: list) -> str:
        r = requests.post(f"{self.base}/detokenize",
                        json={"tokens": tokens}, timeout=30)
        r.raise_for_status()
        return r.json().get("content", "")
    
    def generate(
        self,
        audio: np.ndarray,
        context: str = "",
        force_language: Optional[str] = None,
        assistant_prefix: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """
        与 R2T2ASRModel._generate_streaming_chunk 对接。
        - assistant_prefix 为空且给了 force_language：先注入 "language X<asr_text>"
        让模型首轮就锁定语言，避免猜成 English。
        - 返回纯增量（已剥掉 assistant_prefix）。
        """
        # 首轮防护：没给 prefix 时，至少给个 language tag
        if not assistant_prefix and force_language:
            assistant_prefix = f"language {force_language}<asr_text>"

        messages = []
        if context:
            messages.append({"role": "system", "content": context})

        messages.append({
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": self._wav_to_b64(audio), "format": "wav"},
            }],
        })

        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})

        payload = {
            "messages": messages,
            "chat_template": _ASR_CHAT_TEMPLATE,
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
        }
        # debug_print(f"messages: {messages}")
        r = requests.post(f"{self.base}/v1/chat/completions",
                        json=payload, timeout=self.timeout)
        r.raise_for_status()
        full = r.json()["choices"][0]["message"]["content"]
        if assistant_prefix and full.startswith(assistant_prefix):
            return full[len(assistant_prefix):]
        return full
    
    def transcribe_with_prefix(
        self,
        wav16k: np.ndarray,
        prefix_text: str = "",
        prompt: str = "Transcribe the audio.",
        max_tokens: int = 4096,
    ) -> str:
        """
        发送 [user: 音频+指令] + [assistant: prefix_text]，
        让模型从 prefix_text 之后继续生成。
        """
        messages = [
            {"role": "user", "content": [
                {"type": "input_audio",
                "input_audio": {"data": self._wav_to_b64(wav16k), "format": "wav"}},
                {"type": "text", "text": prompt},
            ]},
        ]
        if prefix_text:
            messages.append({"role": "assistant", "content": prefix_text})
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        r = requests.post(f"{self.base}/v1/chat/completions",
                        json=payload, timeout=self.timeout)
        r.raise_for_status()
        full_content = r.json()["choices"][0]["message"]["content"]
        # llama-server 返回完整 assistant 内容（prefix + 增量），剥掉 prefix
        if prefix_text and full_content.startswith(prefix_text):
            return full_content[len(prefix_text):]
        return full_content
    
    
class LlamaServerStreaming:
    """
    与 R2T2ASRModel.streaming_transcribe 对齐的 llama-server 流式后端。

    prefix 通过 messages[-1]={"role":"assistant","content":prefix} 实现，
    模型从 prefix 之后续写；客户端剥掉 prefix，得到纯增量 delta。
    """

    def __init__(
        self,
        client: "LlamaServerClient",
        context: str = "",
        language: str = "Chinese",
        unfixed_chunk_num: int = 0,
        unfixed_token_num: int = 4,       # 回退长度，含义由 rollback_mode 决定
        chunk_size_sec: float = 0.5,
        prompt_raw: str = "Transcribe the audio.",
        max_audio_sec: float | None = 16.0,
        rollback_mode: str = "char",      # "char"（快）或 "token"（精确）
    ):
        self.client = client
        self.prompt_raw = prompt_raw
        self.force_language = language
        self.unfixed_chunk_num = int(unfixed_chunk_num)
        self.unfixed_token_num = int(unfixed_token_num)
        self.chunk_size_sec = float(chunk_size_sec)
        self.chunk_size_samples = max(1, int(chunk_size_sec * 16000))
        self.max_audio_samples = (
            int(max_audio_sec * 16000) if max_audio_sec else None
        )
        self.rollback_mode = rollback_mode

        self.chunk_id = 0
        self.buffer = np.zeros((0,), dtype=np.float32)
        self.audio_accum = np.zeros((0,), dtype=np.float32)
        self._raw_decoded = ""
        self.language = ""
        self.text = ""

    # ---------- 内部工具 ----------

    def _build_prefix(self) -> str:
        """构造送给模型的 assistant 前缀（含 language X<asr_text>）。"""
        if self.chunk_id < self.unfixed_chunk_num:
            return ""
        if not self._raw_decoded:
            return ""

        if self.rollback_mode == "token":
            tokens = self.client.tokenize(self._raw_decoded)
            k = int(self.unfixed_token_num)
            while True:
                end_idx = max(0, len(tokens) - k)
                cand = (
                    self.client.detokenize(tokens[:end_idx]) if end_idx > 0 else ""
                )
                if "\ufffd" not in cand:
                    return cand
                if end_idx == 0:
                    return ""
                k += 1

        # 字符级：尾部回退 N 字符，保证不切断多字节字符
        s = self._raw_decoded
        n = int(self.unfixed_token_num)
        if n >= len(s):
            return ""
        return s[:-n]

    def _apply_window(self) -> None:
        if (
            self.max_audio_samples
            and self.audio_accum.shape[0] > self.max_audio_samples
        ):
            self.audio_accum = self.audio_accum[-self.max_audio_samples:]

    def _one_step(self) -> str:
        """执行一次完整迭代：构造 prefix → HTTP → 更新状态 → 返回当前 text。"""
        prefix = self._build_prefix()

        delta = self.client.transcribe_with_prefix(
            self.audio_accum,
            prefix_text=prefix,
            prompt=self.prompt_raw,
            max_tokens=4096,
        )
        delta = delta.replace("\ufffd", "")

        if prefix:
            # 拼回完整 raw，用于下一轮 rollback
            self._raw_decoded = prefix + delta
        else:
            # 首轮/无 prefix：模型会自己带出 language X<asr_text>
            self._raw_decoded = delta
            if "<asr_text>" not in self._raw_decoded:
                self._raw_decoded = (
                    f"language {self.force_language}<asr_text>" + self._raw_decoded
                )

        lang, txt = parse_asr_output(
            self._raw_decoded, user_language=self.force_language
        )
        self.language = lang or self.language
        self.text = txt.split("|")[0].split("#")[0]
        self.chunk_id += 1
        return self.text

    # ---------- 公开 API ----------

    def push(self, pcm16k: np.ndarray) -> str:
        x = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        self.buffer = np.concatenate([self.buffer, x])

        while self.buffer.shape[0] >= self.chunk_size_samples:
            chunk = self.buffer[: self.chunk_size_samples]
            self.buffer = self.buffer[self.chunk_size_samples :]
            self.audio_accum = np.concatenate([self.audio_accum, chunk])
            self._apply_window()
            self._one_step()

        return self.text

    def finish(self) -> str:
        if self.buffer.shape[0] == 0:
            return self.text
        self.audio_accum = np.concatenate([self.audio_accum, self.buffer])
        self.buffer = np.zeros((0,), dtype=np.float32)
        self._apply_window()
        self._one_step()
        return self.text
