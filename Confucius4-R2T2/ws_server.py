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
from sanic import Sanic
from sanic.worker.manager import WorkerManager
from sanic import Request, Websocket
import numpy as np
import argparse,traceback
import json
import os,string
import time
from r2t2 import R2T2ASRModel
from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
import asyncio

import logging
import re
from sanic.worker.process import WorkerProcess  # Compatible with Sanic v23.12+; older versions may need: from sanic.worker.manager import WorkerProcess


# Common Chinese and English punctuation characters for hallucination detection
_PUNCT_CHARS = "，。！？、；：,.!?;:~…·\"'()（）《》—-"
_PUNCT_RE = re.compile(r"[%s\s]+" % re.escape(_PUNCT_CHARS))


def _normalize_for_pattern(s: str) -> str:
    """Remove whitespace and punctuation, then lowercase for punctuation-insensitive repeat detection."""
    return _PUNCT_RE.sub("", s).lower()

def detect_and_fix_repetitions(text, threshold=5):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
            
        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                    
                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break
                
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            
            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return ''.join(result)
    
    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text

def detect_hallucination(
        requestId: str,
        text: str,
        pattern_repeat_thresh: int = 5, # Number of repeated tail patterns required to trigger, e.g. "Okay. Okay. Okay." = 3 repeats
        max_pattern_len: int = 50,      # Maximum pattern length to check in characters, covering sentence-level repeats
        tail_check_len: int = 256       # Number of tail characters to check
        ):
    """
    Determine whether text is suspected ASR hallucination (repetition / pattern loop).
    Only checks the last tail_check_len characters.
    Returns (is_hallucination: bool, reason: str)
    """
    if not text:
        return False, ""
    # Only check the tail portion of the text
    text = text[-tail_check_len:]

    # 2. Exact tail pattern loop matching, catches cases like "Okay. Okay. Okay."
    n = len(text)
    for k in range(1, max_pattern_len + 1):
        if n < k * pattern_repeat_thresh:
            continue
        pattern = text[-k:]
        if pattern.strip(_PUNCT_CHARS + " \t") == "":
            continue  # Pure punctuation patterns are already handled by 1b
        ok = True
        for r in range(1, pattern_repeat_thresh):
            if text[-(r + 1) * k:-r * k] != pattern:
                ok = False
                break
        if ok:
            return True, f"tail_pattern:'{pattern}'x{pattern_repeat_thresh}+"

    # 3. Fuzzy tail pattern loop matching after removing punctuation and whitespace, catches noisy variants like "Okay. O kay, okay."
    norm_tail = _normalize_for_pattern(text)
    n2 = len(norm_tail)
    for k in range(3, max_pattern_len + 1):  # Start from 3 because 2-character patterns are too short and can false-positive on English
        if n2 < k * pattern_repeat_thresh:
            continue
        pattern = norm_tail[-k:]
        ok = True
        for r in range(1, pattern_repeat_thresh):
            if norm_tail[-(r + 1) * k:-r * k] != pattern:
                ok = False
                break
        if ok:
            return True, f"tail_pattern_norm:'{pattern}'x{pattern_repeat_thresh}+"

    return False, ""

# Save the original __init__ method
original_init = WorkerProcess.__init__

def patched_init(self, *args, **kwargs):
    # Call the original __init__
    original_init(self, *args, **kwargs)
    # Force daemon to False
    self.daemon = False

# Replace the WorkerProcess __init__ method
WorkerProcess.__init__ = patched_init

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(_handler)
logger.propagate = False

# --- Sanic configuration ---
Sanic.START_METHOD_SET = True
# Alternatively switch directly to the safer 'spawn' mode: Sanic.start_method = "spawn"
Sanic.start_method = "fork"
app = Sanic("onetime_asr_stream_server")
# Disable websocket ping/pong to avoid pong timeouts when synchronous inference blocks the event loop
app.config.WEBSOCKET_PING_INTERVAL = None
app.config.WEBSOCKET_PING_TIMEOUT = None
WorkerManager.THRESHOLD = 300

# --- Global constants and argument parsing ---
YOUDAO_ONETIME_ASR_EOS_STRING = "YOUDAO_ONETIME_ASR_STREAM_EOS"
YOUDAO_TRANSLATE_STREAM_EOS = "YOUDAO_TRANSLATE_STREAM_EOS"
SAMPLING_RATE =16000

current_file_dir = os.path.dirname(os.path.abspath(__file__))
WAV_STORE_DIR = os.path.join(current_file_dir,"wav_tmp_store")
os.makedirs(WAV_STORE_DIR, exist_ok=True)

CHUNK_ASR_SECONDS = 0.16
CHUNK_ASR_SIZE = int(CHUNK_ASR_SECONDS * SAMPLING_RATE)
STEP_MS = int(CHUNK_ASR_SECONDS * 1000)
LOOKAHEAD_MS = 160
UNFIX_TOKEN_NUM = 1
MAX_TOKENS=10

ERROR_MSG_NO_HEADER = "json header is expected"
ERROR_MSG_INVALID_HEADER = "invalid json header"
MAX_SYSTEM_PROMPT_CHARS = 4000


def resolve_system_prompt(header):
    if "system_prompt" not in header:
        return ""

    system_prompt = header["system_prompt"]
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    if len(system_prompt) > MAX_SYSTEM_PROMPT_CHARS:
        raise ValueError(
            f"system_prompt must be at most {MAX_SYSTEM_PROMPT_CHARS} characters"
        )
    return system_prompt.strip()


def redact_stream_request_header(header):
    redacted = dict(header)
    if "secret_key" in redacted:
        redacted["secret_key"] = "[redacted]"
    if "system_prompt" in redacted:
        system_prompt = header["system_prompt"]
        redacted["system_prompt"] = "[redacted]"
        redacted["system_prompt_chars"] = (
            len(system_prompt.strip()) if isinstance(system_prompt, str) else 0
        )
    return redacted


def resolve_qwen_context(header, smooth):
    context_parts = []
    if smooth:
        context_parts.append("Smooth the text")
    system_prompt = resolve_system_prompt(header)
    if system_prompt:
        context_parts.append(system_prompt)
    return "\n".join(context_parts)

# --- Model placeholder ---
asr_model = None # Also create a placeholder for the ASR model

# --- Sanic listener: load models independently in each worker process ---
@app.listener('before_server_start')
async def initialize_models(app):
    """
    This function is called when each worker process starts.
    Loading the model here ensures every process has its own clean, independent model instance.
    """
    logger.info(f"Worker (pid: {os.getpid()}) is initializing models...")

    # Use the global keyword to modify variables in the global scope
    global stream_vad, asr_model

    # 1. Load the VAD model
    vad_config=FireRedStreamVadConfig(
        use_gpu=True,
        smooth_window_size=5,
        speech_threshold=0.4,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=2000,
        min_silence_frame=20,
        chunk_max_frame=30000)
    
    stream_vad = FireRedStreamVad.from_pretrained(args.vad_model_path, vad_config)
    
    logger.info("✅ VAD model initialized successfully")

    # 2. Load the ASR model
    logger.info(f"step2: Start loading ASR model from {args.asr_model_path}")

    # Streaming is vLLM-only and no forced aligner supported.
    asr_model =  R2T2ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=0.95, #0.8
        max_new_tokens=1,
    )

    logger.info("✅ ASR model initialized successfully")

    # Warm up: trigger CUDA kernel compilation, vLLM scheduler initialization, etc.
    logger.info("step3: Start warming up ASR model")
    _sr = SAMPLING_RATE
    _step = int(round(STEP_MS / 1000.0 * _sr))
    _lookhead = int(round(LOOKAHEAD_MS / 1000.0 * _sr))
    for _warmup_lang in ("Chinese", "English"):
        _warmup_state = asr_model.init_streaming_state(
            language=_warmup_lang,
            unfixed_chunk_num=0,
            unfixed_token_num=UNFIX_TOKEN_NUM,
            chunk_size_sec=CHUNK_ASR_SECONDS,
        )
        _warmup_max_new_tokens = max(1, int((_step + _lookhead) / 1280))
        if _warmup_lang == "Chinese":
            _warmup_max_new_tokens = 2 * _warmup_max_new_tokens
        for _ in range(3):
            _dummy = np.zeros(CHUNK_ASR_SIZE, dtype=np.float32)
            asr_model.streaming_transcribe(_dummy, _warmup_state, _warmup_max_new_tokens)
        asr_model.finish_streaming_transcribe(_warmup_state, _warmup_max_new_tokens)
    logger.info("✅ ASR model warmup complete")

    logger.info(f"Worker (pid: {os.getpid()}) model initialization complete.")


def read_pcm(pcm_byte_data, is_wav=False):
    # ... (function code unchanged) ...
    pcm_data = np.frombuffer(pcm_byte_data, dtype=np.int16, offset=44 if is_wav else 0)
    res = pcm_data / (2 ** 15)
    return res.astype(np.float32)

def if_contains_wav_header(byte_array):
    # ... (function code unchanged) ...
    if len(byte_array) < 12: return False
    if byte_array[0:4] != b'RIFF': return False
    if byte_array[8:12] != b'WAVE': return False
    return True

def validate_header(header):
    if "requestId" not in header:
        return False
    return True


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


def remove_punctuation(text):
    """
    Remove all Chinese and English punctuation from a string.
    """
    # English punctuation
    en_punct = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[]^_`{|}~


    # Chinese punctuation (common range)
    cn_punct = "？！＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。〃〄々〆〇〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"


    # Combine all punctuation
    all_punct = en_punct + cn_punct


    # Remove punctuation
    translator = str.maketrans('', '', all_punct)
    return text.translate(translator)


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


# Current active connection count
active_connections = 0
secret_key_list=["test0102"] # only for debug

@app.websocket("/asr_stream_api_v0")
async def asr_stream_api_v0(request: Request, ws: Websocket):
    """
    Use the receiver + queue pattern:
    the receiver coroutine continuously receives data and puts it into the queue,
    while the processor coroutine reads from the queue and runs inference.
    Receiving and inference are decoupled to prevent TCP buffer buildup when
    inference blocks ws.recv().
    """
    global active_connections
    # Ensure the model has been loaded
    if asr_model is None:
        print("Error: ASR model has not been initialized in the current worker process!")
        await ws.close(code=1011, reason="Server model not ready")
        return

    audio_buf = np.ndarray(shape=(0,), dtype=np.float32)

    header_raw = await ws.recv()
    header = json.loads(header_raw)
    logger.info(f"header={redact_stream_request_header(header)}")

    is_first_seg = True
    tmp_audio_pointer = 0
    requestId = None
    total_secs = 0
    last_fixed_asr_text = ""
    no_new_text_chunks = 0
    counted = False
    _req_handler = None  # Per-request log handler

    _STOP = object()  # Sentinel object that tells the processor to exit
    recv_queue: asyncio.Queue = asyncio.Queue()
    RECV_TIMEOUT = 120

    async def receiver():
        """Continuously receive client data and put it into recv_queue."""
        try:
            while True:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.info(f"requestId={requestId}: ws.recv() timeout after {RECV_TIMEOUT}s, closing")
                    break
                await recv_queue.put(data)
                # Exit the receiver after receiving EOS
                if isinstance(data, str) and data == YOUDAO_ONETIME_ASR_EOS_STRING:
                    break
        except Exception as e:
            logger.info(f"requestId={requestId}: receiver exit: {e}")
        finally:
            await recv_queue.put(_STOP)

    async def processor():
        """Read data from recv_queue, run ASR inference, and send results."""
        global active_connections
        nonlocal audio_buf, tmp_audio_pointer, is_first_seg, total_secs
        nonlocal last_fixed_asr_text, counted, requestId, _req_handler, no_new_text_chunks

        try:
            if not validate_header(header):
                logger.error(f"requestId={requestId}:disconnected. error msg: {ERROR_MSG_NO_HEADER}")
                await ws.send(json.dumps({"status": "error", "msg": ERROR_MSG_NO_HEADER}))
                await ws.close()
                return

            requestId = header.get("requestId", None)
            # Create an independent log file for this request
            _req_log_dir = os.path.join(os.path.dirname(__file__), "logs", "requests")
            os.makedirs(_req_log_dir, exist_ok=True)
            _req_log_path = os.path.join(_req_log_dir, f"{requestId}.log")
            _req_handler = logging.FileHandler(_req_log_path, encoding="utf-8")
            _req_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            _logger = logging.getLogger(f"asr_request.{requestId}")
            _logger.setLevel(logging.DEBUG)
            _logger.propagate = False
            _logger.addHandler(_req_handler)
            _logger.info(
                f"requestId={requestId}: header={redact_stream_request_header(header)}"
            )

            use_vad = header.get("use_vad", False)
            language = header.get("language", "zhen")
            if language == "zhen":
                language = None
            secret_key = header.get("secret_key", None)
            infer_mode = header.get("mode", "slow")
            nmt_fast = False
            if infer_mode != "slow":
                nmt_fast = True
            smooth = header.get("smooth", False)
            debug_network = header.get("debug_network", False)
            if debug_network:
                debug_network_text = ""
            if secret_key is None or secret_key not in secret_key_list:
                _logger.info(f"requestId={requestId}: invalid secret_key")
                await ws.close(code=4401, reason="Unauthorized")
                return

            try:
                context = resolve_qwen_context(header, smooth)
            except ValueError as e:
                error_message = str(e)
                _logger.info(f"requestId={requestId}: invalid system_prompt")
                await ws.send(json.dumps({
                    "status": "error",
                    "requestId": f"{requestId}",
                    "msg": error_message,
                }))
                await ws.close(code=1008, reason="invalid system_prompt")
                return

            active_connections += 1
            counted = True
            await ws.send(json.dumps({"status": "connected", "requestId": f"{requestId}", "msg": "", "active_connections": active_connections}))
            _logger.info(f"requestId={requestId}: connected, active_connections={active_connections}")
            _logger.info(f"requestId={requestId}: connected, language={language}, use_vad={use_vad}, nmt_fast={nmt_fast}")

            if args.save_audio:
                WAV_STORE_DIR_requestId = os.path.join(WAV_STORE_DIR, f"{requestId}")
                os.makedirs(WAV_STORE_DIR_requestId, exist_ok=True)
            sr = SAMPLING_RATE
            step = int(round(STEP_MS / 1000.0 * sr))
            lookahead = int(round(LOOKAHEAD_MS / 1000.0 * sr))

            asr_state = asr_model.init_streaming_state(
                context=context,
                language=language,
                unfixed_chunk_num=0,
                unfixed_token_num=UNFIX_TOKEN_NUM,
                chunk_size_sec=CHUNK_ASR_SECONDS,
            )

            if use_vad:
                stream_vad.reset()
                _logger.info(f"requestId={requestId}: stream_vad.reset() done")

            max_new_tokens = max(1, int((step + lookahead) / 1280))
            max_new_tokens_floor = min(32, max(4,2 * int(step / 1280)))
            is_first = True
            total_secs = LOOKAHEAD_MS / 1000.0
            is_halluc = False
            halluc_reason = ""
            recv_time = 0
            total_new_asr_tokens = []
            _recent_cost_ms = []  # Cost of the latest 10 chunks
            avg_cost_ms = 0
            prefix=""
            total_text=""
            ws_closed = False

            while True:
                data = await recv_queue.get()
                if data is _STOP:
                    _logger.info(f"requestId={requestId}: processor got STOP from receiver, exit")
                    break

                recv_time = time.time()

                if isinstance(data, str):
                    if data == YOUDAO_ONETIME_ASR_EOS_STRING:
                        _logger.info(f"requestId {requestId}: client sent EOS. closing connection.")
                        first_max_new_tokens = max(1, int((step + lookahead) / 1280))
                        text= asr_model.finish_streaming_transcribe(
                            asr_state, first_max_new_tokens
                        )
                        text = text.split("|")[0]
                        if len(text) > len(last_fixed_asr_text):
                            new_asr_text = text[len(last_fixed_asr_text):]
                            text = last_fixed_asr_text + new_asr_text
                            #out_msg = {"text": text, "reset": True}
                            out_msg = {"text": new_asr_text, "reset": True}
                        else:
                            # out_msg = {"text": last_fixed_asr_text, "reset": True}
                            out_msg = {"text": "", "reset": True}
                        out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                        _logger.info(f"requestId={requestId}:finish! out_str={out_str}")
                        await ws.close()
                        break
                    else:
                        _logger.info(f"requestId={requestId}:Received message: {data}")
                        continue

                if is_first_seg:
                    is_wav = if_contains_wav_header(data)
                    audio_seg = read_pcm(data, is_wav)
                else:
                    audio_seg = read_pcm(data)

                # Used to quickly test network latency
                if debug_network:
                    debug_network_text += "Testing network latency..."
                    out_msg = {"text": debug_network_text, "reset": False, "asr_cost_ms": 0, "total_cost_ms": round((time.time() - recv_time) * 1000, 1)}
                    out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                    try:
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                    except Exception as e:
                        _logger.warning(f"requestId={requestId}: ws.send failed (connection closed?): {e}")
                        break
                    continue

                audio_buf = np.concatenate((audio_buf, audio_seg))
                audio_buf_len = len(audio_buf)
                if is_first and audio_buf_len < int(step + lookahead):
                    await ws.send(json.dumps({}))
                    continue

                #-------- Start iterating over audio_buf -------#
                while audio_buf_len - tmp_audio_pointer >= CHUNK_ASR_SIZE:
                    if is_first:
                        asr_state.chunk_size_sec = (step + lookahead) / sr
                        chunk_size_samples = int(round(float(asr_state.chunk_size_sec) * sr))
                        asr_state.chunk_size_samples = max(1, chunk_size_samples)
                        audio_seg = audio_buf[tmp_audio_pointer:tmp_audio_pointer + step + lookahead]
                        is_first = False
                        total_secs += (step + lookahead) / sr
                        tmp_audio_pointer = step + lookahead
                    else:
                        asr_state.chunk_size_sec = CHUNK_ASR_SECONDS
                        chunk_size_samples = int(round(float(asr_state.chunk_size_sec) * sr))
                        asr_state.chunk_size_samples = max(1, chunk_size_samples)
                        audio_seg = audio_buf[tmp_audio_pointer:tmp_audio_pointer + CHUNK_ASR_SIZE]
                        tmp_audio_pointer += CHUNK_ASR_SIZE
                        total_secs += CHUNK_ASR_SECONDS

                    # Streaming VAD
                    speech_ended_in_this_chunk = False
                    vad_cost_ms = 0.0
                    finish_cost_ms = 0.0
                    if True or use_vad:
                        audio_seg_int16 = (audio_seg * 32768).astype(np.int16)
                        t0 = time.time()
                        chunk_results = stream_vad.detect_chunk(audio_seg_int16)
                        for r in chunk_results:
                            if r.is_speech_end:
                                speech_ended_in_this_chunk = True
                                _logger.info(
                                    f"requestId={requestId}: VAD speech_end at frame {r.speech_end_frame}")
                        vad_cost_ms = round((time.time() - t0) * 1000, 1)

                    t0 = time.time()
                    text, fixed_asr_text = asr_model.streaming_transcribe(
                        audio_seg, asr_state, int(max_new_tokens), nmt_fast
                    )
                    asr_cost_ms = round((time.time() - t0) * 1000, 1)
                    if len(fixed_asr_text) > len(last_fixed_asr_text):
                        new_asr_text = fixed_asr_text[len(last_fixed_asr_text):]
                        fixed_asr_text = last_fixed_asr_text + new_asr_text
                        out_msg = {"text": new_asr_text, "asr_cost_ms": asr_cost_ms, "reset": False}
                        last_fixed_asr_text = fixed_asr_text

                        new_asr_tokens = split_text_to_tokens(new_asr_text)
                        for asr_word in new_asr_tokens:
                            total_new_asr_tokens.append(asr_word)
                            if len(total_new_asr_tokens) > MAX_TOKENS:
                                total_new_asr_tokens.pop(0)
                        max_new_tokens=max(1, int(step / 1280))
                        no_new_text_chunks = 0
                    else:
                        out_msg = {"text": "", "asr_cost_ms": asr_cost_ms, "reset": False}
                        if not is_last_token_chinese(total_new_asr_tokens):
                            max_new_tokens = max_new_tokens + 0.5
                        else:
                            max_new_tokens=max(1, int(step / 1280))
                        no_new_text_chunks += 1

                    if is_last_token_chinese(total_new_asr_tokens):
                        max_new_tokens = 2 * max_new_tokens

                    max_new_tokens = min(max_new_tokens_floor, max_new_tokens) 

                    total_text = last_fixed_asr_text

                    # Hallucination detection
                    t0 = time.time()
                    cur_halluc, cur_reason = detect_hallucination(requestId, total_text)
                    if cur_halluc:
                        is_halluc = True
                        halluc_reason = cur_reason

                    if speech_ended_in_this_chunk and use_vad:
                        seg_final_text = asr_model.finish_streaming_transcribe(
                            asr_state, max(1, int((step + lookahead) / 1280))
                        )
                        asr_state = asr_model.init_streaming_state(
                            context=context,
                            language=language,
                            unfixed_chunk_num=0,
                            unfixed_token_num=UNFIX_TOKEN_NUM,
                            chunk_size_sec=CHUNK_ASR_SECONDS,
                        )
                        if len(seg_final_text) > len(last_fixed_asr_text):
                            seg_new_asr_text = seg_final_text[len(last_fixed_asr_text):]
                            seg_final_text = last_fixed_asr_text + seg_new_asr_text
                        else:
                            seg_new_asr_text = ""
                        total_text = seg_final_text

                        out_msg = {"text": seg_new_asr_text, "reset": True, "asr_cost_ms": asr_cost_ms}
                        last_fixed_asr_text = ""
                    finish_cost_ms = round((time.time() - t0) * 1000, 1)
                    out_msg['total_cost_ms'] = vad_cost_ms + asr_cost_ms + finish_cost_ms
                    _recent_cost_ms.append(out_msg['total_cost_ms'])
                    if len(_recent_cost_ms) > 10:
                        _recent_cost_ms.pop(0)
                    avg_cost_ms = round(sum(_recent_cost_ms) / len(_recent_cost_ms), 1)
                    lag_ms = round((time.time() - recv_time) * 1000, 1)
                    out_msg['lag_ms'] = lag_ms
                    out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                    try:
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                    except Exception as e:
                        traceback.print_exc()
                        _logger.warning(f"requestId={requestId}: ws.send failed (connection closed?): {e}")
                        ws_closed = True
                        break
                    _logger.info(f"requestId={requestId}:out_str={out_str},total_secs={total_secs},lag_ms={lag_ms},avg_cost_ms={avg_cost_ms},infer_mode={infer_mode},max_new_tokens={max_new_tokens} for next,total_text={total_text}")

                #-------- End iterating over audio_buf -------#
                if ws_closed:
                    break
                if len(audio_buf) != 0:
                    audio_buf = audio_buf[tmp_audio_pointer:]
                tmp_audio_pointer = 0

                if is_first_seg:
                    is_first_seg = False

                SOFT_RESET_SEC = 1.0 * 60
                HARD_RESET_SEC = 1.5 * 60
                did_reset = False
                if is_halluc:
                    _logger.info(
                        f"requestId={requestId}: hallucination reset, reason={halluc_reason}, "
                        f"total_secs={total_secs:.1f}s, discard current state"
                    )                    
                    text = asr_model.finish_streaming_transcribe(
                        asr_state, max(1, int((step + lookahead) / 1280))
                    )
                    text = text.split("|")[0]
                    if len(text) > len(last_fixed_asr_text):
                        text = last_fixed_asr_text + text[len(last_fixed_asr_text):]
                        out_msg = {"text": text, "reset": True}
                    else:
                        out_msg = {"text": last_fixed_asr_text, "reset": True}
                    asr_state = asr_model.init_streaming_state(
                        context=context,
                        language=language,
                        unfixed_chunk_num=0,
                        unfixed_token_num=UNFIX_TOKEN_NUM,
                        chunk_size_sec=CHUNK_ASR_SECONDS,
                    )
                    did_reset = True
                else:
                    if total_secs >= HARD_RESET_SEC:
                        _logger.info(
                            f"requestId={requestId}: HARD reset, total_secs={total_secs:.1f}s, "
                            f"no VAD silence detected, force reset"
                        )
                        text = asr_model.finish_streaming_transcribe(
                            asr_state, max(1, int((step + lookahead) / 1280))
                        )
                        text = text.split("|")[0]
                        if len(text) > len(last_fixed_asr_text):
                            new_asr_text = text[len(last_fixed_asr_text):]
                            text = last_fixed_asr_text + new_asr_text
                            out_msg = {"text": new_asr_text, "reset": True}
                        else:
                            out_msg = {"text": "", "reset": True}
                        asr_state = asr_model.init_streaming_state(
                            context=context,
                            language=language,
                            unfixed_chunk_num=0,
                            unfixed_token_num=UNFIX_TOKEN_NUM,
                            chunk_size_sec=CHUNK_ASR_SECONDS,
                        )                        
                        _rolling_audio_buf = np.zeros(0, dtype=np.float32)
                        did_reset = True
                    elif not use_vad and total_secs > SOFT_RESET_SEC:
                        _logger.info(
                            f"requestId={requestId}: HARD reset, total_secs={total_secs:.1f}s, "
                            f"no VAD silence detected, force reset,avg_cost_ms={avg_cost_ms}"
                        )                        
                        if speech_ended_in_this_chunk:
                            text = asr_model.finish_streaming_transcribe(
                                asr_state, max(1, int((step + lookahead) / 1280))
                            )
                            text = text.split("|")[0]
                            if len(text) > len(last_fixed_asr_text):
                                new_asr_text = text[len(last_fixed_asr_text):]
                                text = last_fixed_asr_text + new_asr_text
                                out_msg = {"text": new_asr_text, "reset": True}
                            else:
                                out_msg = {"text": "", "reset": True}
                            asr_state = asr_model.init_streaming_state(
                                context=context,
                                language=language,
                                unfixed_chunk_num=0,
                                unfixed_token_num=UNFIX_TOKEN_NUM,
                                chunk_size_sec=CHUNK_ASR_SECONDS,
                            )
                            did_reset = True

                if did_reset:
                    total_secs = 0.0
                    is_halluc = False
                    halluc_reason = ""
                    out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                    last_fixed_asr_text = ""
                    total_text=""
                    no_new_text_chunks = 0
                    try:
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                        _logger.info(f"requestId={requestId}:out_str={out_str},did_reset={did_reset}")
                    except Exception as e:
                        _logger.info(f"requestId={requestId}: ws.send(reset) failed: {e}")
                        break

                try:
                    await ws.send(json.dumps({}))
                except Exception as e:
                    _logger.info(f"requestId={requestId}: ws.send({{}}) failed: {e}")
                    break

        except asyncio.CancelledError as e:
            _logger.exception(f"requestId={requestId}: processor cancelled: {e}")
        except Exception as e:
            _logger.exception(f"requestId={requestId}: error:{e}")

    # Start receiver and processor
    receiver_task = asyncio.create_task(receiver())
    try:
        await processor()
    except Exception as e:
        logger.exception(f"requestId={requestId}: youdao_asr_stream_api_new error: {e}")
    finally:
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass
        if counted:
            active_connections -= 1
        logger.info(f"requestId={requestId}: connection closed, active_connections={active_connections}")
        # Clean up the per-request log handler
        if _req_handler is not None:
            _req_logger = logging.getLogger(f"asr_request.{requestId}")
            _req_logger.removeHandler(_req_handler)
            _req_handler.close()
        await ws.close()

@app.websocket("/asr_stream_api_v1")
async def asr_stream_api_v1(request: Request, ws: Websocket):
    """
    Use the receiver + queue pattern:
    the receiver coroutine continuously receives data and puts it into the queue,
    while the processor coroutine reads from the queue and runs inference.
    Receiving and inference are decoupled to prevent TCP buffer buildup when
    inference blocks ws.recv().
    """
    global active_connections
    # Ensure the model has been loaded
    if asr_model is None:
        print("Error: ASR model has not been initialized in the current worker process!")
        await ws.close(code=1011, reason="Server model not ready")
        return

    audio_buf = np.ndarray(shape=(0,), dtype=np.float32)

    header_raw = await ws.recv()
    header = json.loads(header_raw)
    logger.info(f"header={redact_stream_request_header(header)}")

    is_first_seg = True
    tmp_audio_pointer = 0
    requestId = None
    total_secs = 0
    last_fixed_asr_text = ""
    no_new_text_chunks = 0
    counted = False
    _req_handler = None  # Per-request log handler

    _STOP = object()  # Sentinel object that tells the processor to exit
    recv_queue: asyncio.Queue = asyncio.Queue()
    RECV_TIMEOUT = 120

    async def receiver():
        """Continuously receive client data and put it into recv_queue."""
        try:
            while True:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.info(f"requestId={requestId}: ws.recv() timeout after {RECV_TIMEOUT}s, closing")
                    break
                await recv_queue.put(data)
                # Exit the receiver after receiving EOS
                if isinstance(data, str) and data == YOUDAO_ONETIME_ASR_EOS_STRING:
                    break
        except Exception as e:
            logger.info(f"requestId={requestId}: receiver exit: {e}")
        finally:
            await recv_queue.put(_STOP)

    async def processor():
        """Read data from recv_queue, run ASR inference, and send results."""
        global active_connections
        nonlocal audio_buf, tmp_audio_pointer, is_first_seg, total_secs
        nonlocal last_fixed_asr_text, counted, requestId, _req_handler, no_new_text_chunks

        try:
            if not validate_header(header):
                logger.error(f"requestId={requestId}:disconnected. error msg: {ERROR_MSG_NO_HEADER}")
                await ws.send(json.dumps({"status": "error", "msg": ERROR_MSG_NO_HEADER}))
                await ws.close()
                return

            requestId = header.get("requestId", None)
            # Create an independent log file for this request
            _req_log_dir = os.path.join(os.path.dirname(__file__), "logs", "requests")
            os.makedirs(_req_log_dir, exist_ok=True)
            _req_log_path = os.path.join(_req_log_dir, f"{requestId}.log")
            _req_handler = logging.FileHandler(_req_log_path, encoding="utf-8")
            _req_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            _logger = logging.getLogger(f"asr_request.{requestId}")
            _logger.setLevel(logging.DEBUG)
            _logger.propagate = False
            _logger.addHandler(_req_handler)
            _logger.info(
                f"requestId={requestId}: header={redact_stream_request_header(header)}"
            )

            use_vad = header.get("use_vad", False)
            language = header.get("language", "zhen")
            if language == "zhen":
                language = None
            secret_key = header.get("secret_key", None)
            infer_mode = header.get("mode", "slow")
            nmt_fast = False
            if infer_mode != "slow":
                nmt_fast = True
            smooth = header.get("smooth", False)
            debug_network = header.get("debug_network", False)
            if debug_network:
                debug_network_text = ""
            if secret_key is None or secret_key not in secret_key_list:
                _logger.info(f"requestId={requestId}: invalid secret_key")
                await ws.close(code=4401, reason="Unauthorized")
                return

            try:
                context = resolve_qwen_context(header, smooth)
            except ValueError as e:
                error_message = str(e)
                _logger.info(f"requestId={requestId}: invalid system_prompt")
                await ws.send(json.dumps({
                    "status": "error",
                    "requestId": f"{requestId}",
                    "msg": error_message,
                }))
                await ws.close(code=1008, reason="invalid system_prompt")
                return

            active_connections += 1
            counted = True
            await ws.send(json.dumps({"status": "connected", "requestId": f"{requestId}", "msg": "", "active_connections": active_connections}))
            _logger.info(f"requestId={requestId}: connected, active_connections={active_connections}")
            _logger.info(f"requestId={requestId}: connected, language={language}, use_vad={use_vad}, nmt_fast={nmt_fast}")

            if args.save_audio:
                WAV_STORE_DIR_requestId = os.path.join(WAV_STORE_DIR, f"{requestId}")
                os.makedirs(WAV_STORE_DIR_requestId, exist_ok=True)
            sr = SAMPLING_RATE
            step = int(round(STEP_MS / 1000.0 * sr))
            lookahead = int(round(LOOKAHEAD_MS / 1000.0 * sr))

            asr_state = asr_model.init_streaming_state(
                context=context,
                language=language,
                unfixed_chunk_num=0,
                unfixed_token_num=UNFIX_TOKEN_NUM,
                chunk_size_sec=CHUNK_ASR_SECONDS,
            )

            if use_vad:
                stream_vad.reset()
                _logger.info(f"requestId={requestId}: stream_vad.reset() done")

            max_new_tokens = max(1, int((step + lookahead) / 1280))
            max_new_tokens_floor = min(32, max(4,2 * int(step / 1280)))
            is_first = True
            total_secs = LOOKAHEAD_MS / 1000.0
            is_halluc = False
            halluc_reason = ""
            recv_time = 0
            total_new_asr_tokens = []
            _recent_cost_ms = []  # Cost of the latest 10 chunks
            avg_cost_ms = 0
            prefix=""
            total_text=""
            ws_closed = False
            
            while True:
                data = await recv_queue.get()
                if data is _STOP:
                    _logger.info(f"requestId={requestId}: processor got STOP from receiver, exit")
                    break

                recv_time = time.time()

                if isinstance(data, str):
                    if data == YOUDAO_ONETIME_ASR_EOS_STRING:
                        _logger.info(f"requestId {requestId}: client sent EOS. closing connection.")
                        first_max_new_tokens = max(1, int((step + lookahead) / 1280))
                        text= asr_model.finish_streaming_transcribe_no_reset(
                            asr_state, first_max_new_tokens
                        )
                        text = text.split("|")[0]
                        if len(text) > len(last_fixed_asr_text):
                            new_asr_text = text[len(last_fixed_asr_text):]
                            text = last_fixed_asr_text + text[len(last_fixed_asr_text):]
                            out_msg = {"text": new_asr_text, "reset": True}
                        else:
                            out_msg = {"text": "", "reset": True}
                        total_text = text
                        out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                        _logger.info(f"requestId={requestId}:finish! out_str={out_str},total_text={total_text}")
                        await ws.close()
                        break
                    else:
                        _logger.info(f"requestId={requestId}:Received message: {data}")
                        continue

                if is_first_seg:
                    is_wav = if_contains_wav_header(data)
                    audio_seg = read_pcm(data, is_wav)
                else:
                    audio_seg = read_pcm(data)

                # Used to quickly test network latency
                if debug_network:
                    debug_network_text += "Testing network latency..."
                    out_msg = {"text": debug_network_text, "reset": False, "asr_cost_ms": 0, "total_cost_ms": round((time.time() - recv_time) * 1000, 1)}
                    out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                    try:
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                    except Exception as e:
                        _logger.warning(f"requestId={requestId}: ws.send failed (connection closed?): {e}")
                        break
                    continue

                audio_buf = np.concatenate((audio_buf, audio_seg))
                audio_buf_len = len(audio_buf)
                if is_first and audio_buf_len < int(step + lookahead):
                    await ws.send(json.dumps({}))
                    continue

                #-------- Start iterating over audio_buf -------#
                while audio_buf_len - tmp_audio_pointer >= CHUNK_ASR_SIZE:
                    if is_first:
                        asr_state.chunk_size_sec = (step + lookahead) / sr
                        chunk_size_samples = int(round(float(asr_state.chunk_size_sec) * sr))
                        asr_state.chunk_size_samples = max(1, chunk_size_samples)
                        audio_seg = audio_buf[tmp_audio_pointer:tmp_audio_pointer + step + lookahead]
                        is_first = False
                        total_secs += (step + lookahead) / sr
                        tmp_audio_pointer = step + lookahead
                    else:
                        asr_state.chunk_size_sec = CHUNK_ASR_SECONDS
                        chunk_size_samples = int(round(float(asr_state.chunk_size_sec) * sr))
                        asr_state.chunk_size_samples = max(1, chunk_size_samples)
                        audio_seg = audio_buf[tmp_audio_pointer:tmp_audio_pointer + CHUNK_ASR_SIZE]
                        tmp_audio_pointer += CHUNK_ASR_SIZE
                        total_secs += CHUNK_ASR_SECONDS

                    # Streaming VAD
                    speech_ended_in_this_chunk = False
                    vad_cost_ms = 0.0
                    finish_cost_ms = 0.0
                    if True or use_vad:
                        audio_seg_int16 = (audio_seg * 32768).astype(np.int16)
                        t0 = time.time()
                        chunk_results = stream_vad.detect_chunk(audio_seg_int16)
                        for r in chunk_results:
                            if r.is_speech_end:
                                speech_ended_in_this_chunk = True
                                _logger.info(
                                    f"requestId={requestId}: VAD speech_end at frame {r.speech_end_frame}")
                        vad_cost_ms = round((time.time() - t0) * 1000, 1)

                    t0 = time.time()
                    text, fixed_asr_text = asr_model.streaming_transcribe_no_reset(
                        audio_seg, asr_state, int(max_new_tokens), nmt_fast
                    )
                    asr_cost_ms = round((time.time() - t0) * 1000, 1)
                    if len(fixed_asr_text) > len(last_fixed_asr_text):
                        new_asr_text = fixed_asr_text[len(last_fixed_asr_text):]
                        fixed_asr_text = last_fixed_asr_text + new_asr_text
                        out_msg = {"text": new_asr_text, "asr_cost_ms": asr_cost_ms, "reset": False}
                        last_fixed_asr_text = fixed_asr_text

                        new_asr_tokens = split_text_to_tokens(new_asr_text)
                        for asr_word in new_asr_tokens:
                            total_new_asr_tokens.append(asr_word)
                            if len(total_new_asr_tokens) > MAX_TOKENS:
                                total_new_asr_tokens.pop(0)
                        max_new_tokens=max(1, int(step / 1280))
                        no_new_text_chunks = 0
                    else:
                        out_msg = {"text": "", "asr_cost_ms": asr_cost_ms, "reset": False}
                        if not is_last_token_chinese(total_new_asr_tokens):
                            max_new_tokens = max_new_tokens + 0.5
                        else:
                            max_new_tokens=max(1, int(step / 1280))
                        no_new_text_chunks += 1

                    total_text = last_fixed_asr_text

                    if is_last_token_chinese(total_new_asr_tokens):
                        max_new_tokens = 2 * max_new_tokens

                    max_new_tokens = min(max_new_tokens_floor, max_new_tokens) 

                    # Hallucination detection
                    t0 = time.time()
                    cur_halluc, cur_reason = detect_hallucination(requestId, total_text)
                    if cur_halluc:
                        is_halluc = True
                        halluc_reason = cur_reason

                    if speech_ended_in_this_chunk and use_vad:
                        # seg_final_text = asr_model.finish_streaming_transcribe_no_reset(
                        #     asr_state, max(1, int((step + lookahead) / 1280))
                        # )
                        seg_final_text = asr_state.text
                        seg_final_text = seg_final_text.split("|")[0]
                        asr_state = asr_model.init_streaming_state(
                            context=context,
                            language=language,
                            unfixed_chunk_num=0,
                            unfixed_token_num=UNFIX_TOKEN_NUM,
                            chunk_size_sec=CHUNK_ASR_SECONDS,
                        )
                        if len(seg_final_text) > len(last_fixed_asr_text):
                            seg_new_asr_text = seg_final_text[len(last_fixed_asr_text):]
                            seg_final_text = last_fixed_asr_text + seg_new_asr_text
                        else:
                            seg_new_asr_text = ""
                        total_text = seg_final_text

                        out_msg = {"text": seg_new_asr_text, "reset": True, "asr_cost_ms": asr_cost_ms}
                        last_fixed_asr_text = ""
                    finish_cost_ms = round((time.time() - t0) * 1000, 1)
                    out_msg['total_cost_ms'] = vad_cost_ms + asr_cost_ms + finish_cost_ms
                    _recent_cost_ms.append(out_msg['total_cost_ms'])
                    if len(_recent_cost_ms) > 10:
                        _recent_cost_ms.pop(0)
                    avg_cost_ms = round(sum(_recent_cost_ms) / len(_recent_cost_ms), 1)
                    lag_ms = round((time.time() - recv_time) * 1000, 1)
                    out_msg['lag_ms'] = lag_ms
                    out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                    try:
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                    except Exception as e:
                        traceback.print_exc()
                        _logger.warning(f"requestId={requestId}: ws.send failed (connection closed?): {e}")
                        ws_closed = True
                        break
                    _logger.info(f"requestId={requestId}:out_str={out_str},total_secs={total_secs},lag_ms={lag_ms},avg_cost_ms={avg_cost_ms},infer_mode={infer_mode},max_new_tokens={max_new_tokens} for next,total_text={total_text}")

                #-------- End iterating over audio_buf -------#
                if ws_closed:
                    break
                if len(audio_buf) != 0:
                    audio_buf = audio_buf[tmp_audio_pointer:]
                tmp_audio_pointer = 0

                if is_first_seg:
                    is_first_seg = False

                did_reset = False
                if is_halluc:
                    _logger.info(
                        f"requestId={requestId}: hallucination reset, reason={halluc_reason}, "
                        f"total_secs={total_secs:.1f}s, discard current state"
                    )                    
                    text = asr_model.finish_streaming_transcribe_no_reset(
                        asr_state, max(1, int((step + lookahead) / 1280))
                    )
                    text = text.split("|")[0]
                    if len(text) > len(last_fixed_asr_text):
                        new_asr_text = text[len(last_fixed_asr_text):]
                        text = last_fixed_asr_text + new_asr_text
                        out_msg = {"text": new_asr_text, "reset": True}
                    else:
                        out_msg = {"text": "", "reset": True}
                    asr_state = asr_model.init_streaming_state(
                        context=context,
                        language=language,
                        unfixed_chunk_num=0,
                        unfixed_token_num=UNFIX_TOKEN_NUM,
                        chunk_size_sec=CHUNK_ASR_SECONDS,
                    )
                    did_reset = True

                if did_reset:
                    total_secs = 0.0
                    is_halluc = False
                    halluc_reason = ""
                    out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                    last_fixed_asr_text = "" 
                    total_text = ""
                    no_new_text_chunks = 0
                    try:
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                        _logger.info(f"requestId={requestId}:out_str={out_str},did_reset={did_reset}")
                    except Exception as e:
                        _logger.info(f"requestId={requestId}: ws.send(reset) failed: {e}")
                        break

                try:
                    await ws.send(json.dumps({}))
                except Exception as e:
                    _logger.info(f"requestId={requestId}: ws.send({{}}) failed: {e}")
                    break

        except asyncio.CancelledError as e:
            _logger.exception(f"requestId={requestId}: processor cancelled: {e}")
        except Exception as e:
            _logger.exception(f"requestId={requestId}: error:{e}")

    # Start receiver and processor
    receiver_task = asyncio.create_task(receiver())
    try:
        await processor()
    except Exception as e:
        logger.exception(f"requestId={requestId}: youdao_asr_stream_api_new error: {e}")
    finally:
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass
        if counted:
            active_connections -= 1
        logger.info(f"requestId={requestId}: connection closed, active_connections={active_connections}")
        # Clean up the per-request log handler
        if _req_handler is not None:
            _req_logger = logging.getLogger(f"asr_request.{requestId}")
            _req_logger.removeHandler(_req_handler)
            _req_handler.close()
        await ws.close()


def args_parser():
    parser = argparse.ArgumentParser(description='asr stream server')
    # ... (argument parsing code unchanged) ...
    parser.add_argument('-p', '--port', default="8272", dest="port", help="Server websocket port")
    parser.add_argument('-ip', '--ip', default="0.0.0.0", dest="ip", help="Server websocket ip")
    parser.add_argument('-s', '--save_audio', action="store_true", dest="save_audio", help="Whether to save audio for debugging purpose")
    parser.add_argument(
        '-m', '--asr_model_path',
        default=os.environ.get("ASR_MODEL_PATH"),
        dest="asr_model_path",
        help="Path or HF repo id of the Confucius4-R2T2 ASR checkpoint. "
             "Falls back to the ASR_MODEL_PATH environment variable if not provided.",
    )
    parser.add_argument(
        '--vad_model_path',
        default=os.path.join(current_file_dir, "checkpoints", "vad", "Stream-VAD"),
        dest="vad_model_path",
        help="Path to the FireRedVAD Stream-VAD model. Defaults to checkpoints/vad/Stream-VAD "
             "(see the README for how to download it).",
    )
    args = parser.parse_args()
    if not args.asr_model_path:
        parser.error(
            "ASR model path is required. Pass it via --asr_model_path "
            "or set the ASR_MODEL_PATH environment variable."
        )
    return args

args = args_parser()

# --- Main program entry point ---
if __name__ == '__main__':
    app.run(host=args.ip, port=int(args.port), single_process=True)
