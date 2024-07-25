import os
import json
import time
from pydantic import BaseModel, Field
import requests
from pydub import AudioSegment
from pydub.generators import Sine

import numpy as np
import shutil

# docker run -d --name chattts -p 9100:8000 -v /tmp/audio:/audio ghcr.io/ultrasev/chattts:latest

def read_text(txtfile):
    with open(txtfile, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            yield index, line.strip()

def create_silence_audio(duration=1000):
    sample_rate = 44100
    silence = AudioSegment.silent(duration=duration, frame_rate=sample_rate)
    return silence


def concatenate_audio(audio_a, audio_b, output_file):
    combined_audio = audio_a + audio_b
    combined_audio.export(output_file, format="wav")


def merge_audio(file1, file2, output_file):
    if not os.path.exists(file1):
        silence_audio = create_silence_audio(duration=1000)
        audio_b = AudioSegment.from_file(file2)
        concatenate_audio(silence_audio, audio_b, output_file)
        return

    audio_a = AudioSegment.from_file(file1)
    silence_audio = create_silence_audio(duration=0.5)
    audio_b = AudioSegment.from_file(file2)
    combined_audio = audio_a + silence_audio + audio_b
    combined_audio.export(output_file, format="wav")


class TextToSpeechReq(BaseModel):
    text: str = Field(default="")
    output_path: str = Field(default="")
    seed: int = Field(default=232)


class WebException(Exception):
    def __init__(self, status_code, message="Web request failed"):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Status Code: {self.status_code})"


def post_tts(line, audio_file):
    url = "http://localhost:9100/tts"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = TextToSpeechReq(
        text=line,
        output_path=f"/audio/{audio_file}",
        seed=232
    ).json()
    print(f"{payload=}")
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        raise WebException(response.status_code, response.text)
    return response.json()


def delete_files(folder):
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            os.remove(item_path)


delete_files("./outputs/")
for idx, line in read_text("read.txt"):
    print(f"{line}")
    if line=='':
        continue
    post_tts(line, "current.wav")
    prev_file = f"outputs/prev.wav"
    target_file = f"outputs/read.wav"
    merge_audio(prev_file, f"/tmp/audio/current.wav", target_file)
    shutil.copyfile(target_file, prev_file)

