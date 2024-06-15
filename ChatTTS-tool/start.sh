docker rm -f chattts
docker run -d --name chattts --gpus all -p 9100:8000 -v /tmp/audio:/audio ghcr.io/ultrasev/chattts:latest