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

import argparse
import numpy as np
import librosa
import re,string
from r2t2 import R2T2ASRModel

def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return wav.astype(np.float32, copy=False)
    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)

def read_wav_bytes_with_librosa(file_path):
    """
    Read audio with librosa and return the audio data and sample rate.
    """
    # 1. Load audio with librosa as a NumPy array
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)

    return audio_data,sample_rate

def remove_punctuation(text):
    """
    Remove all Chinese and English punctuation from a string.
    """
    # English punctuation
    en_punct = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    # Chinese punctuation (common range)
    cn_punct = "？！＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。〃〄々〆〇〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"

    # Combine all punctuation
    all_punct = en_punct + cn_punct

    # Remove punctuation
    translator = str.maketrans('', '', all_punct)
    return text.translate(translator)

def split_text_to_tokens(text: str) -> list:
    """
    Split text into a token list: split Chinese by character, English by word,
    and ignore punctuation.
    Example: "你好,world。测试" -> ["你", "好", "world", "测", "试"]
    """
    text = remove_punctuation(text)
    tokens = []
    # Match consecutive English words or a single non-English character
    pattern = re.compile(r'[a-zA-Z]+|[^a-zA-Z]')
    for match in pattern.finditer(text):
        token = match.group()
        if token.strip():  # Ignore whitespace characters
            tokens.append(token)
    return tokens

def is_last_token_chinese(new_asr_tokens: list) -> bool:
    """
    Determine whether the last element in new_asr_tokens is Chinese or English.
    Return True for Chinese and False for English.
    Return False if the list is empty.
    """
    if not new_asr_tokens:
        return False
    last = new_asr_tokens[-1]
    # Treat the token as Chinese if it contains any Chinese character
    for ch in last:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def run_streaming(asr: R2T2ASRModel, 
                wav16k: np.ndarray, 
                step_ms: int, 
                chunk_size_sec: float, 
                unfixed_token_num: int, 
                lookahead_ms: int, 
                language=None, 
                context="") -> None:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))
    lookhead = int(round(lookahead_ms / 1000.0 * sr))
    
    state = asr.init_streaming_state(
        context=context,
        language=language,
        unfixed_chunk_num=0,
        unfixed_token_num=unfixed_token_num,
        chunk_size_sec=chunk_size_sec,
    )    

    pos = 0
    call_id = 0

    new_text = ""
    last_text_tmp=""

    max_new_tokens=max(1, int((step + lookhead) / 1280))
    first_max_new_tokens = max_new_tokens
    is_first = True
    total_new_asr_tokens=[]

    max_new_tokens_floor = min(32, max(4,2 * int(step / 1280)))
    while pos < wav16k.shape[0]:
        if is_first == True:
            seg = wav16k[pos : pos + step + lookhead]
            is_first = False
            state.chunk_size_sec = (step + lookhead) / sr
            chunk_size_samples = int(round(float(state.chunk_size_sec) * sr))
            state.chunk_size_samples = max(1, chunk_size_samples)
        else:
            seg = wav16k[pos : pos + step]
            state.chunk_size_sec = chunk_size_sec
            chunk_size_samples = int(round(float(state.chunk_size_sec) * sr))
            state.chunk_size_samples = max(1, chunk_size_samples)
        pos += seg.shape[0]
        call_id += 1

        _,text= asr.streaming_transcribe(seg, state, int(max_new_tokens))
        text=text.split("|")[0]

        print(f"text={text}")
        if len(text) > len(last_text_tmp):
            last_text_tmp = text
            max_new_tokens=max(1, int(step / 1280))
        else:
            if not is_last_token_chinese(total_new_asr_tokens):
                max_new_tokens = max_new_tokens + 1
            else:
                max_new_tokens=max(1, int(step / 1280))
        
        if is_last_token_chinese(total_new_asr_tokens):
            max_new_tokens = 2 * max_new_tokens

        max_new_tokens = min(max_new_tokens_floor, max_new_tokens) 
   
        if seg.shape[0] / sr < chunk_size_sec:
            break

    asr.finish_streaming_transcribe(state, first_max_new_tokens)
    state.text=state.text.split("|")[0]
    return state.text

def init_stream_asr_model(model_path):
    """Initialize the ASR model."""
    asr = R2T2ASRModel.LLM(
        model=model_path,
        gpu_memory_utilization=0.4,
        max_new_tokens=4, # set a small value for streaming
    )
    return asr

def init_onetime_asr_model(model_path):
    asr = R2T2ASRModel.LLM(
        model=model_path,
        gpu_memory_utilization=0.5,
        forced_aligner=None,
        max_inference_batch_size=32,
        max_new_tokens=4096,
    )
    return asr

def run_onetime(asr: R2T2ASRModel, wav16k: np.ndarray, language=None) -> None:
    results = asr.transcribe(
        audio=[(wav16k, 16000)],
        language=[language],
        return_time_stamps=False,
    )
    return results[0].text

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Streaming Example (single file)")
    p.add_argument("--audio", type=str, required=True, help="Path to the audio file.")
    p.add_argument("--model_path", type=str, required=True, help="Path to the ASR model.")
    p.add_argument("--infer_mode", type=str, default="stream_vllm",
                help="stream_vllm or onetime_vllm")
    p.add_argument("--language", type=str, default=None,
                   help="Language hint (e.g. Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish，Arabi). None if not set.")
    p.add_argument("--chunk_size_ms", type=int, default=160,
                   help="Chunk size in milliseconds (default: 160).")
    p.add_argument("--lookahead_ms", "--lookahead_ms", dest="lookahead_ms", type=int, default=160,
                   help="Lookahead size in milliseconds (default: 160). "
                        "`--lookahead_ms` is accepted as a deprecated alias.")
    p.add_argument("--unfixed_token_num", type=int, default=1,
                   help="Number of unfixed tokens (default: 1).")
    p.add_argument("--context", type=str, default="", help="Context / hotword hint.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Load audio
    print(f"Loading audio from {args.audio} ...")
    wav,sr = read_wav_bytes_with_librosa(args.audio)
    wav16k = _resample_to_16k(wav, sr)

    chunk_size_sec = args.chunk_size_ms / 1000.0
    language = args.language if args.language and args.language != "None" else None

    # Run streaming ASR
    # Load model
    if args.infer_mode == "stream_vllm":
        print(f"Loading model from {args.model_path} ...")
        asr = init_stream_asr_model(model_path=args.model_path)    
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
        print(f"\n===== Final Result (Streaming)=====")
        print(f"final_result={result}")
    if args.infer_mode == "onetime_vllm":
        print(f"Loading model from {args.model_path} ...")
        asr = init_onetime_asr_model(model_path=args.model_path)    
        result = run_onetime(
            asr, 
            wav16k,
            language=language,
        )
        print(f"\n===== Final Result (Onetime)=====")
        print(f"final_result={result}")