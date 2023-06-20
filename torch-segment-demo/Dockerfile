FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip
RUN pip3 install torch torchvision

# initialize app
WORKDIR /app
COPY . /app

CMD ["python3", "main.py"]