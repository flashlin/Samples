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
import asyncio,os
import argparse
import websockets
import json
import numpy as np
import uuid
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CHUNK_SIZE_MS = 160          # Send one frame every 160 ms
LANGUAGE = "zhen"            # zhen/Chinese/English
USE_VAD = False              # Whether to enable server-side VAD; disabled by default. Enabling VAD may reduce quality.

DEFAULT_WEBSOCKET_URI = "ws://localhost:8272/asr_stream_api_v1"
YOUDAO_ONETIME_ASR_EOS_STRING = "YOUDAO_ONETIME_ASR_STREAM_EOS"

# Built-in sample audio shipped with the repository.
DEFAULT_AUDIO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", "test.wav"
)

async def send_audio_new(websocket, input_file_path, block_size, num_channels):
    index = 0
    for audio_block in sf.blocks(input_file_path, dtype="int16",
                                 blocksize=block_size, always_2d=False):
        if audio_block.ndim == 1:
            audio_block = audio_block.reshape(-1, 1)

        if len(audio_block) < block_size:
            pad_n = int(SAMPLE_RATE * 0.5)
            zeros = np.zeros((pad_n, num_channels), dtype="int16")
            audio_block = np.vstack([audio_block, zeros])

        await websocket.send(audio_block.tobytes())
        index += 1
    
    await websocket.send(YOUDAO_ONETIME_ASR_EOS_STRING)
    print(f"Sent {index} audio frames; waiting for the final result")


async def receive_audio_new(websocket, asr_save_file, audio_id):
    out_total_txt = []
    while True:
        try:
            server_out = await asyncio.wait_for(websocket.recv(), timeout=10)
        except asyncio.TimeoutError:
            print("Timeout waiting for response")
            break
        except websockets.ConnectionClosed as e:
            if e.code == 1000:
                print(f"Server closed connection normally (code={e.code})")
            else:
                print(f"Server closed connection abnormally (code={e.code}, reason={e.reason})")
                raise
            break

        if not isinstance(server_out, str):
            print(f"Warning: got non-str message: {type(server_out)}")
            continue

        server_out_json = json.loads(server_out)
        if 'status' not in server_out_json:
            continue

        msg = server_out_json.get('msg', {})
        if not isinstance(msg, dict):
            continue
        asr_txt = msg.get('text', '')

        out_total_txt.append(asr_txt)

        tmp_asr_txt = "".join(out_total_txt)
        print(f"tmp_asr_txt={tmp_asr_txt}", flush=True)

    final_asr_txt = "".join(out_total_txt)

    with open(asr_save_file, "a", encoding="utf-8") as file:
        final_line = f"{audio_id}\t{final_asr_txt}\n"
        file.write(final_line)


async def test_service_ws_from_file_new(audio_path, asr_save_file, audio_id, websocket_uri):
    requestId = str(uuid.uuid4())
    print(f"requestId={requestId}")

    in_file_info = sf.SoundFile(audio_path)
    print(in_file_info)

    num_channels = in_file_info.channels
    sample_rate = in_file_info.samplerate
    block_size = int(CHUNK_SIZE_MS / 1000 * sample_rate)

    async with websockets.connect(websocket_uri, ping_interval=None) as websocket:
        metadata = {
            "channels": num_channels,
            "sample_rate": sample_rate,
            "requestId": requestId,
            "language": LANGUAGE,
            "use_vad": USE_VAD,
            "secret_key": "test0102",
            "mode": "slow"
        }
        await websocket.send(json.dumps(metadata))

        send_task = asyncio.create_task(
            send_audio_new(websocket, audio_path, block_size, num_channels))
        recv_task = asyncio.create_task(
            receive_audio_new(websocket, asr_save_file, audio_id))
        await asyncio.gather(send_task, recv_task)


def test_one_audio(audio_path, asr_save_file, audio_id, websocket_uri, max_retries=1):
    for attempt in range(1 + max_retries):
        try:
            asyncio.run(test_service_ws_from_file_new(
                audio_path=audio_path, asr_save_file=asr_save_file,
                audio_id=audio_id, websocket_uri=websocket_uri))
            return
        except Exception as e:
            if attempt < max_retries:
                print(f"audio_id={audio_id}: Attempt {attempt+1} failed ({e}); retrying...")
            else:
                print(f"audio_id={audio_id}: Still failed after {max_retries} retries ({e}); skipping")


def parse_args():
    parser = argparse.ArgumentParser(description="WebSocket ASR client")
    parser.add_argument(
        "--uri", "-u",
        default=os.environ.get("ASR_WS_URI", DEFAULT_WEBSOCKET_URI),
        help=f"WebSocket URI, e.g. ws://host:port/asr_stream_api_v1 (default: {DEFAULT_WEBSOCKET_URI}, "
             f"or env ASR_WS_URI)",
    )
    parser.add_argument(
        "--audio", "-a",
        default=DEFAULT_AUDIO_PATH,
        help=f"Path to input audio file (default: built-in sample {DEFAULT_AUDIO_PATH})",
    )
    parser.add_argument(
        "--save", "-s",
        default="service_ws_test",
        help="File to append ASR results to",
    )
    parser.add_argument(
        "--audio-id",
        default=None,
        help="Identifier written next to the result; defaults to the audio file basename",
    )
    return parser.parse_args()


def test_client():
    args = parse_args()
    audio_id = args.audio_id or os.path.basename(args.audio)
    print(f"connecting to {args.uri}")
    test_one_audio(
        audio_path=args.audio,
        asr_save_file=args.save,
        audio_id=audio_id,
        websocket_uri=args.uri,
    )

if __name__ == "__main__":
    test_client()
    exit(0)
