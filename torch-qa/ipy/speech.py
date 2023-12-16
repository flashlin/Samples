import os
import pyaudio
# import audioop
import wave
import numpy as np

ambient_detected = False
speech_volume = 100


def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    print("可用的錄音設備：")
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Device ID: {i}, Name: {device_info.get('name')}")
    p.terminate()
    print("=========================")

def get_rms(data):
    rms = np.sqrt(np.mean(np.square(data))) / 32767
    return rms


def live_speech(transcribe_fn, wait_time=10):
    global ambient_detected
    global speech_volume

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    print("open audio")
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    recording = False
    frames_recorded = 0

    print("start..")
    while True:
        frames_recorded += 1
        data = stream.read(CHUNK)
        # rms = audioop.rms(data, 2)
        rms = get_rms(data)

        if not ambient_detected:
            if frames_recorded < 40:
                if frames_recorded == 1:
                    print("Detecting ambient noise...")
                if frames_recorded > 5:
                    if speech_volume < rms:
                        speech_volume = rms
                continue
            elif frames_recorded == 40:
                print("Listening...")
                speech_volume = speech_volume * 3
                ambient_detected = True

        if rms > speech_volume:
            recording = True
            frames_recorded = 0
        elif recording and frames_recorded > wait_time:
            recording = False

            wf = wave.open("audio.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            result = transcribe_fn("audio.wav")
            # result = whisper_model.transcribe(
            #     "audio.wav",
            #     fp16=False
            # )
            os.remove("audio.wav")

            # yield result["text"].strip()
            print(f"{result=}")
            # yield result
            frames = []

        if recording:
            frames.append(data)

    # TODO: do these when breaking from generator
    stream.stop_stream()
    stream.close()
    audio.terminate()


def transcribe(data):
    return ""


if __name__ == '__main__':
    list_audio_devices()
    live_speech(transcribe)
