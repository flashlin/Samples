docker rm -f chattts
docker run -d --name chattts --gpus all -p 9100:8000 \
    -v /tmp/audio:/audio ghcr.io/ultrasev/chattts:latest

#-v /mnt/d/VDisk/tts_models/models--2Noise--ChatTTS:chattts:/root/.cache/huggingface/hub/models--2Noise--ChatTTS/blobs

# docker exec -it chattts /bin/bash
