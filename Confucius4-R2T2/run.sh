#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
HF_MODEL_DIR="$SCRIPT_DIR/models/Confucius4-R2T2"
MLX_MODEL_DIR="$SCRIPT_DIR/models/Confucius4-R2T2-MLX-8bit"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "正在建立虛擬環境並安裝依賴..."
    uv venv "$SCRIPT_DIR/.venv" --python 3.12
    uv pip install --python "$VENV_PYTHON" mlx-audio qwen3-asr-mlx huggingface_hub sounddevice librosa soundfile
fi

if [ ! -f "$MLX_MODEL_DIR/model.safetensors" ]; then
    if [ ! -f "$HF_MODEL_DIR/model.safetensors" ]; then
        echo "正在下載 netease-youdao/Confucius4-R2T2 權重..."
        "$SCRIPT_DIR/.venv/bin/hf" download netease-youdao/Confucius4-R2T2 --local-dir "$HF_MODEL_DIR"
    fi
    echo "正在將 Confucius4-R2T2 轉換為 8-bit MLX 格式..."
    "$VENV_PYTHON" -m mlx_audio.convert \
        --hf-path "$HF_MODEL_DIR" \
        --mlx-path "$MLX_MODEL_DIR" \
        --quantize \
        --q-bits 8 \
        --q-group-size 64 \
        --model-domain stt
fi

exec "$VENV_PYTHON" "$SCRIPT_DIR/mlx_r2t2.py" --model_path "$MLX_MODEL_DIR" "$@"
