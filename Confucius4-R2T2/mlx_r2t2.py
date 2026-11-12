# coding=utf-8
"""
Confucius4-R2T2 MLX Streaming & Offline ASR on Apple Silicon (Mac).

Supports:
1. Real-time microphone streaming ASR (default) — prints stable-prefix (append-only)
   transcription directly to the console as you speak.
2. Audio file streaming / offline transcription via `--audio <file.wav>`.
"""

import argparse
import queue
import re
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import mlx.core as mx
import numpy as np
from mlx_audio.stt import load as load_mlx_stt

SAMPLE_RATE = 16000
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "

_EN2ZH_PUNCT = {
    ",": "，",
    ".": "。",
    "!": "！",
    "?": "？",
    ";": "；",
    ":": "：",
    "(": "（",
    ")": "）",
}
_ZH2EN_PUNCT = {v: k for k, v in _EN2ZH_PUNCT.items()}
_ALL_PUNCT_PAT = re.compile(
    r"[,\.!?;:()\uff0c\u3002\uff01\uff1f\uff1b\uff1a\uff08\uff09]"
)


def _normalize_punct_by_context(text: str) -> str:
    """Normalize punctuation based on the preceding character (Chinese vs ASCII)."""

    def _replace(m):
        punct = m.group()
        pos = m.start()
        prev_char = ""
        for i in range(pos - 1, -1, -1):
            if not text[i].isspace():
                prev_char = text[i]
                break
        if not prev_char:
            return punct
        if "\u4e00" <= prev_char <= "\u9fff":
            return _EN2ZH_PUNCT.get(punct, punct)
        elif prev_char.isascii() and (prev_char.isalnum() or prev_char in "\"'"):
            return _ZH2EN_PUNCT.get(punct, punct)
        return punct

    return _ALL_PUNCT_PAT.sub(_replace, text)


def parse_asr_output(
    raw: str, user_language: Optional[str] = None
) -> Tuple[str, str]:
    """Parse Qwen3-ASR / R2T2 raw output into (language, text)."""
    if not raw:
        return "", ""
    s = str(raw).rstrip() if user_language == "English" else str(raw).strip()
    if not s:
        return "", ""
    if user_language:
        return user_language, s

    if _ASR_TEXT_TAG in s:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        return "", s.strip()

    meta_lower = meta_part.lower()
    if "language none" in meta_lower:
        t = text_part.strip()
        return ("", t) if t else ("", "")

    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if line.lower().startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX) :].strip()
            if val:
                lang = val.capitalize()
            break
    return lang, text_part.strip()


def remove_punctuation(text: str) -> str:
    en_punct = string.punctuation
    cn_punct = "？！＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。〃〄々〆〇〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
    return text.translate(str.maketrans("", "", en_punct + cn_punct))


def split_text_to_tokens(text: str) -> List[str]:
    text = remove_punctuation(text)
    pattern = re.compile(r"[a-zA-Z]+|[^a-zA-Z]")
    return [m.group() for m in pattern.finditer(text) if m.group().strip()]


def is_last_token_chinese(tokens: List[str]) -> bool:
    if not tokens:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in tokens[-1])


@dataclass
class MLXStreamingState:
    """Streaming state for Confucius4-R2T2 on MLX."""

    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int
    chunk_id: int = 0
    buffer: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    audio_accum: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    context: str = ""
    force_language: Optional[str] = None
    language: str = ""
    text: str = ""
    _raw_decoded: str = ""
    chunk_records: List[Tuple[int, str]] = field(default_factory=list)
    committed_prefix: str = ""


class MLXR2T2ASR:
    """MLX wrapper for Confucius4-R2T2 supporting stable-prefix streaming."""

    def __init__(self, model_path: str):
        self.model_path = str(model_path)
        self.model = load_mlx_stt(self.model_path)
        self.tokenizer = self.model._tokenizer
        self.eos_token_ids = self.model._eos_token_ids()

    def warmup(self, language: Optional[str] = "Chinese") -> None:
        dummy = np.zeros(int(0.32 * SAMPLE_RATE), dtype=np.float32)
        state = self.init_streaming_state(
            language=language, unfixed_chunk_num=0, unfixed_token_num=1, chunk_size_sec=0.32
        )
        self.streaming_transcribe(dummy, state, max_new_tokens=2)
        mx.clear_cache()

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 0,
        unfixed_token_num: int = 1,
        chunk_size_sec: float = 0.16,
    ) -> MLXStreamingState:
        force_language = None
        if language is not None and str(language).strip() not in ("", "None", "auto"):
            force_language = str(language).strip()
        chunk_size_samples = max(1, int(round(float(chunk_size_sec) * SAMPLE_RATE)))
        return MLXStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=chunk_size_samples,
            context=context or "",
            force_language=force_language,
        )

    def _build_prompt_with_prefix(
        self,
        num_audio_tokens: int,
        force_language: Optional[str],
        context: str,
        prefix: str,
    ) -> mx.array:
        system_content = f"{context}\n" if context else ""
        if force_language is not None:
            assistant_prefix = f"language {force_language}<asr_text>{prefix}"
        else:
            assistant_prefix = prefix

        prompt = (
            f"<|im_start|>system\n{system_content}<|im_end|>\n"
            f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * num_audio_tokens}<|audio_end|><|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_prefix}"
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        return mx.array(input_ids)

    def _generate_greedy_with_prefix(
        self,
        audio_accum: np.ndarray,
        force_language: Optional[str],
        context: str,
        prefix: str,
        max_new_tokens: int,
    ) -> str:
        if max_new_tokens <= 0 or audio_accum.shape[0] == 0:
            return ""

        input_features, feature_attention_mask, num_audio_tokens = (
            self.model._preprocess_audio(audio_accum)
        )
        audio_features = self.model.get_audio_features(
            input_features, feature_attention_mask
        )
        mx.eval(audio_features)

        input_ids = self._build_prompt_with_prefix(
            num_audio_tokens=num_audio_tokens,
            force_language=force_language,
            context=context,
            prefix=prefix,
        )
        inputs_embeds = self.model._build_inputs_embeds(input_ids, audio_features)
        mx.eval(inputs_embeds)

        cache = self.model.make_cache()
        logits = self.model._forward_with_embeds(inputs_embeds, cache=cache)
        mx.eval(logits)

        generated_tokens: List[int] = []
        for step_idx in range(max_new_tokens):
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            if next_token in self.eos_token_ids:
                break
            generated_tokens.append(next_token)

            # Early stop if Confucius4-R2T2 emits the '|' <Wait> boundary token
            piece = self.tokenizer.decode([next_token], skip_special_tokens=False)
            if "|" in piece:
                break

            if step_idx < max_new_tokens - 1:
                next_embed = self.model.model.embed_tokens(mx.array([[next_token]]))
                logits = self.model._forward_with_embeds(next_embed, cache=cache)
                mx.eval(logits)

        del input_features, feature_attention_mask, audio_features, inputs_embeds, cache
        mx.clear_cache()

        if not generated_tokens:
            return ""
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _trim_tokens(self, raw_decoded: str, k: int) -> str:
        cur_ids = self.tokenizer.encode(raw_decoded, add_special_tokens=False)
        while True:
            end_idx = max(0, len(cur_ids) - k)
            trimmed = (
                self.tokenizer.decode(cur_ids[:end_idx], skip_special_tokens=False)
                if end_idx > 0
                else ""
            )
            if "\ufffd" not in trimmed:
                return trimmed
            if end_idx == 0:
                return ""
            k += 1

    def streaming_transcribe(
        self,
        pcm16k: np.ndarray,
        state: MLXStreamingState,
        max_new_tokens: int = 4,
        rollback_punctuation: bool = False,
    ) -> Tuple[str, str]:
        """
        Feed 16kHz mono PCM samples and run incremental Confucius4-R2T2 decoding.
        Returns (state.text, fixed_text).
        """
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        fixed_text = ""
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]

            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

            # Build prefix with Confucius4-R2T2 rollback strategy
            if state.chunk_id < state.unfixed_chunk_num:
                prefix = ""
            else:
                state._raw_decoded = state._raw_decoded.split("|")[0]
                if rollback_punctuation:
                    _punct = "，。！？、；：.!?;:"
                    raw_stripped = state._raw_decoded.strip()
                    k = (
                        0
                        if (raw_stripped and raw_stripped[-1] in _punct)
                        else int(state.unfixed_token_num)
                    )
                else:
                    k = int(state.unfixed_token_num)
                prefix = self._trim_tokens(state._raw_decoded, k)

            prefix = prefix.split("|")[0]

            gen_text = self._generate_greedy_with_prefix(
                audio_accum=state.audio_accum,
                force_language=state.force_language,
                context=state.context,
                prefix=prefix,
                max_new_tokens=max_new_tokens,
            )
            gen_text = _normalize_punct_by_context(gen_text).replace("\ufffd", "")
            state._raw_decoded = (prefix + gen_text) if prefix else gen_text

            lang, txt = parse_asr_output(
                state._raw_decoded, user_language=state.force_language
            )
            if state.force_language == "Chinese" or lang == "Chinese":
                state._raw_decoded = re.sub(
                    r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])",
                    "",
                    state._raw_decoded,
                )
                lang, txt = parse_asr_output(
                    state._raw_decoded, user_language=state.force_language
                )

            if _ASR_TEXT_TAG in state._raw_decoded:
                state._raw_decoded = (
                    state._raw_decoded.split(_ASR_TEXT_TAG)[0] + _ASR_TEXT_TAG + txt
                )
            else:
                state._raw_decoded = txt

            state._raw_decoded = state._raw_decoded.split("|")[0]

            if rollback_punctuation:
                _punct = "，。！？、；：,.!?;:"
                raw_stripped = state._raw_decoded.strip()
                k = (
                    0
                    if (raw_stripped and raw_stripped[-1] in _punct)
                    else int(state.unfixed_token_num)
                )
            else:
                k = int(state.unfixed_token_num)

            if (
                _ASR_TEXT_TAG in state._raw_decoded
                and state._raw_decoded.split(_ASR_TEXT_TAG)[1] == ""
            ):
                k = 0

            fixed_text = self._trim_tokens(state._raw_decoded, k)
            if _ASR_TEXT_TAG in fixed_text:
                _, fixed_text = fixed_text.split(_ASR_TEXT_TAG, 1)
            fixed_text = fixed_text.split("|")[0]

            if _ASR_TEXT_TAG not in state._raw_decoded and state.force_language is None:
                state.text = ""
                fixed_text = ""
                continue

            state.language = lang
            state.text = txt.split("|")[0]
            state.chunk_id += 1

        return state.text, fixed_text

    def finish_streaming_transcribe(
        self,
        state: MLXStreamingState,
        max_new_tokens: int = 8,
    ) -> str:
        """Flush any remaining audio in buffer and commit final text without rollback."""
        if state.buffer is not None and state.buffer.shape[0] > 0:
            tail = state.buffer
            state.buffer = np.zeros((0,), dtype=np.float32)
            if state.audio_accum.shape[0] == 0:
                state.audio_accum = tail
            else:
                state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

        if state.audio_accum.shape[0] == 0:
            return state.text

        if state.chunk_id < state.unfixed_chunk_num:
            prefix = ""
        else:
            state._raw_decoded = state._raw_decoded.split("|")[0]
            prefix = self._trim_tokens(
                state._raw_decoded, int(state.unfixed_token_num)
            )
        prefix = prefix.split("|")[0]

        gen_text = self._generate_greedy_with_prefix(
            audio_accum=state.audio_accum,
            force_language=state.force_language,
            context=state.context,
            prefix=prefix,
            max_new_tokens=max_new_tokens,
        )
        gen_text = _normalize_punct_by_context(gen_text).replace("\ufffd", "")
        state._raw_decoded = (prefix + gen_text) if prefix else gen_text
        state._raw_decoded = state._raw_decoded.split("|")[0]
        lang, txt = parse_asr_output(
            state._raw_decoded, user_language=state.force_language
        )
        state.language = lang
        state.text = txt.split("|")[0]
        state.chunk_id += 1
        return state.text


def run_file_streaming(
    asr: MLXR2T2ASR,
    audio_path: str,
    chunk_size_ms: int = 160,
    lookahead_ms: int = 160,
    unfixed_token_num: int = 1,
    language: Optional[str] = "Chinese",
    context: str = "",
) -> str:
    """Run Confucius4-R2T2 streaming simulation on an audio file and print incremental output."""
    print(f"Loading audio file: {audio_path}")
    wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    wav = wav.astype(np.float32, copy=False)

    step = int(round(chunk_size_ms / 1000.0 * SAMPLE_RATE))
    lookahead = int(round(lookahead_ms / 1000.0 * SAMPLE_RATE))
    chunk_size_sec = chunk_size_ms / 1000.0

    state = asr.init_streaming_state(
        context=context,
        language=language,
        unfixed_chunk_num=0,
        unfixed_token_num=unfixed_token_num,
        chunk_size_sec=chunk_size_sec,
    )

    pos = 0
    is_first = True
    last_fixed_text = ""
    total_new_asr_tokens: List[str] = []
    max_new_tokens = max(1, int((step + lookahead) / 1280))
    first_max_new_tokens = max_new_tokens
    max_new_tokens_floor = min(32, max(4, 2 * int(step / 1280)))

    print("\n--- Streaming Transcription ---")
    t_start = time.time()
    while pos < wav.shape[0]:
        if is_first:
            seg = wav[pos : pos + step + lookahead]
            is_first = False
            state.chunk_size_sec = (step + lookahead) / SAMPLE_RATE
            state.chunk_size_samples = max(
                1, int(round(state.chunk_size_sec * SAMPLE_RATE))
            )
        else:
            seg = wav[pos : pos + step]
            state.chunk_size_sec = chunk_size_sec
            state.chunk_size_samples = max(
                1, int(round(state.chunk_size_sec * SAMPLE_RATE))
            )
        pos += seg.shape[0]

        _, fixed_text = asr.streaming_transcribe(
            seg, state, max_new_tokens=int(max_new_tokens)
        )
        fixed_text = fixed_text.split("|")[0]

        if len(fixed_text) > len(last_fixed_text):
            delta = fixed_text[len(last_fixed_text) :]
            sys.stdout.write(delta)
            sys.stdout.flush()
            last_fixed_text = fixed_text
            for tok in split_text_to_tokens(delta):
                total_new_asr_tokens.append(tok)
                if len(total_new_asr_tokens) > 10:
                    total_new_asr_tokens.pop(0)
            max_new_tokens = max(1, int(step / 1280))
        else:
            if not is_last_token_chinese(total_new_asr_tokens):
                max_new_tokens += 1
            else:
                max_new_tokens = max(1, int(step / 1280))

        if is_last_token_chinese(total_new_asr_tokens):
            max_new_tokens = 2 * max_new_tokens
        max_new_tokens = min(max_new_tokens_floor, max_new_tokens)

        if seg.shape[0] / SAMPLE_RATE < chunk_size_sec:
            break

    final_text = asr.finish_streaming_transcribe(
        state, max_new_tokens=max(4, first_max_new_tokens)
    )
    if len(final_text) > len(last_fixed_text):
        sys.stdout.write(final_text[len(last_fixed_text) :])
        sys.stdout.flush()
    elapsed = time.time() - t_start
    print(f"\n\n[Completed in {elapsed:.2f}s | Audio duration: {len(wav)/SAMPLE_RATE:.2f}s]")
    return final_text


def run_microphone_streaming(
    asr: MLXR2T2ASR,
    chunk_size_ms: int = 320,
    lookahead_ms: int = 160,
    unfixed_token_num: int = 1,
    language: Optional[str] = "Chinese",
    context: str = "",
    silence_rms_threshold: float = 0.008,
    silence_reset_sec: float = 1.2,
    max_segment_sec: float = 15.0,
    device: Optional[int] = None,
) -> None:
    """Capture live microphone audio and print stable-prefix ASR results in the console."""
    import sounddevice as sd

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[Audio status: {status}]", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    step_samples = int(round(chunk_size_ms / 1000.0 * SAMPLE_RATE))
    lookahead_samples = int(round(lookahead_ms / 1000.0 * SAMPLE_RATE))
    chunk_size_sec = chunk_size_ms / 1000.0

    def new_state() -> MLXStreamingState:
        return asr.init_streaming_state(
            context=context,
            language=language,
            unfixed_chunk_num=0,
            unfixed_token_num=unfixed_token_num,
            chunk_size_sec=chunk_size_sec,
        )

    state = new_state()
    is_first = True
    last_fixed_text = ""
    total_new_asr_tokens: List[str] = []
    max_new_tokens = max(2, int((step_samples + lookahead_samples) / 1280))
    max_new_tokens_floor = min(32, max(6, 2 * int(step_samples / 1280)))

    speech_active = False
    silence_duration = 0.0
    pending_pcm = np.zeros((0,), dtype=np.float32)

    print("==========================================================")
    print("🎙️  Confucius4-R2T2 MLX 即時語音辨識已啟動 (按 Ctrl+C 結束)")
    print(
        f"   模型: {asr.model_path}\n"
        f"   語言: {language or 'Auto'} | 分塊: {chunk_size_ms}ms | U={unfixed_token_num}"
    )
    print("==========================================================\n")
    sys.stdout.write("🗣️  請開始說話: ")
    sys.stdout.flush()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * 0.08),  # 80ms callback blocks
            device=device,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            while True:
                chunk = audio_queue.get()
                while not audio_queue.empty():
                    try:
                        chunk = np.concatenate([chunk, audio_queue.get_nowait()], axis=0)
                    except queue.Empty:
                        break

                pending_pcm = np.concatenate([pending_pcm, chunk], axis=0)
                needed = (step_samples + lookahead_samples) if is_first else step_samples
                if pending_pcm.shape[0] < needed:
                    continue

                seg = pending_pcm[:needed]
                pending_pcm = pending_pcm[needed:]
                seg_sec = seg.shape[0] / SAMPLE_RATE
                rms = float(np.sqrt(np.mean(seg**2)))

                if rms < silence_rms_threshold:
                    silence_duration += seg_sec
                    if not speech_active:
                        # Keep a tiny pre-roll of 160ms so speech onset isn't clipped
                        pending_pcm = seg[-lookahead_samples:] if lookahead_samples > 0 else np.zeros((0,), dtype=np.float32)
                        continue
                else:
                    speech_active = True
                    silence_duration = 0.0

                if is_first:
                    is_first = False
                    state.chunk_size_sec = (step_samples + lookahead_samples) / SAMPLE_RATE
                    state.chunk_size_samples = max(
                        1, int(round(state.chunk_size_sec * SAMPLE_RATE))
                    )
                else:
                    state.chunk_size_sec = chunk_size_sec
                    state.chunk_size_samples = max(
                        1, int(round(state.chunk_size_sec * SAMPLE_RATE))
                    )

                _, fixed_text = asr.streaming_transcribe(
                    seg, state, max_new_tokens=int(max_new_tokens)
                )
                fixed_text = fixed_text.split("|")[0]

                if len(fixed_text) > len(last_fixed_text):
                    delta = fixed_text[len(last_fixed_text) :]
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    last_fixed_text = fixed_text
                    for tok in split_text_to_tokens(delta):
                        total_new_asr_tokens.append(tok)
                        if len(total_new_asr_tokens) > 10:
                            total_new_asr_tokens.pop(0)
                    max_new_tokens = max(2, int(step_samples / 1280))
                else:
                    if not is_last_token_chinese(total_new_asr_tokens):
                        max_new_tokens += 1
                    else:
                        max_new_tokens = max(2, int(step_samples / 1280))

                if is_last_token_chinese(total_new_asr_tokens):
                    max_new_tokens = 2 * max_new_tokens
                max_new_tokens = min(max_new_tokens_floor, max_new_tokens)

                accum_sec = state.audio_accum.shape[0] / SAMPLE_RATE
                # End of utterance on silence OR max segment duration reached
                if (speech_active and silence_duration >= silence_reset_sec) or (
                    accum_sec >= max_segment_sec
                ):
                    final_text = asr.finish_streaming_transcribe(
                        state, max_new_tokens=8
                    )
                    final_text = final_text.split("|")[0]
                    if len(final_text) > len(last_fixed_text):
                        sys.stdout.write(final_text[len(last_fixed_text) :])
                        sys.stdout.flush()
                    if last_fixed_text or final_text:
                        sys.stdout.write("\n🗣️  ")
                        sys.stdout.flush()

                    # Reset for next utterance
                    state = new_state()
                    is_first = True
                    last_fixed_text = ""
                    total_new_asr_tokens.clear()
                    speech_active = False
                    silence_duration = 0.0
                    mx.clear_cache()

    except KeyboardInterrupt:
        print("\n\n👋 已停止語音辨識。")


def parse_args():
    default_model = str(
        Path(__file__).resolve().parent / "models" / "Confucius4-R2T2-MLX-8bit"
    )
    parser = argparse.ArgumentParser(
        description="Confucius4-R2T2 MLX Real-Time Streaming ASR on Mac"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model,
        help="Path to the converted MLX Confucius4-R2T2 model directory.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Optional audio file path to transcribe. If omitted, uses live microphone input.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        help="Language hint (e.g. Chinese, English, or None for auto-detect). Default: Chinese.",
    )
    parser.add_argument(
        "--chunk_size_ms",
        type=int,
        default=320,
        help="Streaming chunk size in ms (default: 320ms for mic, 160ms recommended for file).",
    )
    parser.add_argument(
        "--lookahead_ms",
        type=int,
        default=160,
        help="Initial lookahead in ms (default: 160).",
    )
    parser.add_argument(
        "--unfixed_token_num",
        type=int,
        default=1,
        help="Provisional token rollback U (1 for quality, 0 for latency). Default: 1.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Optional context / hotword prompt.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device index for microphone.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    language = (
        None
        if (args.language is None or args.language.lower() in ("none", "auto", ""))
        else args.language
    )

    print(f"⏳ 正在載入 MLX 模型: {args.model_path} ...")
    asr = MLXR2T2ASR(args.model_path)
    asr.warmup(language=language)
    print("✅ 模型載入與預熱完成！")

    if args.audio:
        run_file_streaming(
            asr=asr,
            audio_path=args.audio,
            chunk_size_ms=args.chunk_size_ms,
            lookahead_ms=args.lookahead_ms,
            unfixed_token_num=args.unfixed_token_num,
            language=language,
            context=args.context,
        )
    else:
        run_microphone_streaming(
            asr=asr,
            chunk_size_ms=args.chunk_size_ms,
            lookahead_ms=args.lookahead_ms,
            unfixed_token_num=args.unfixed_token_num,
            language=language,
            context=args.context,
            device=args.device,
        )


if __name__ == "__main__":
    main()
