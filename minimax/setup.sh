#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# setup.sh - 環境與模型一鍵初始化腳本 (Mac Apple Silicon / MPS 專用)
# 負責安裝：
# 1. Python 3.12 虛擬環境 (.venv) 與 comfy-cli、ffmpeg (imageio-ffmpeg) 等工具
# 2. ComfyUI 最新版 (支援原生 MiniMax H3 0.43.0+)
# 3. Custom Nodes:
#    - city96/ComfyUI-GGUF
#    - Video-Rebirth/ComfyUI-HyperFlow-Flash (6步生成 MiniMax H3 Flash 工作流)
#    - numz/ComfyUI-SeedVR2_VideoUpscaler (SeedVR2 3B 影片放大)
# 4. 模型權重下載 (支援斷點續傳):
#    - minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf (HyperFlow H3 Ref2VA Flash 參考圖生片版)
#    - qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors (MiniMax H3 Text Encoder)
#    - minimax_h3_video_vae_fp16.safetensors (Video VAE)
#    - minimax_h3_audio_vae_fp32.safetensors (Audio VAE)
#    - seedvr2_ema_3b_fp16.safetensors & ema_vae_fp16.safetensors (SeedVR2 3B 放大模型)
#    - Sulphur 3: s3_w4a8.safetensors、qwen3vl_32b_minimax_h3_w4a8_int8down.safetensors、s3_corrector.safetensors
# 5. Sulphur 3 H3 校正器 custom node (S3_ComfyUI.zip) 與 workflows/ 內的 ComfyUI 工作流程 (給 gen.sh 使用)
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# HuggingFace token：只讀取 .env 中的 HF_TOKEN= 一行 (不 source 整個 .env)，可用 HF_ENV_FILE 覆寫路徑
source "$ROOT_DIR/lib/load_hf_token.sh"

echo "========================================================"
echo "🚀 開始執行 MiniMax H3 HyperFlow Flash + SeedVR2 環境安裝"
echo "📁 專案目錄: $ROOT_DIR"
echo "========================================================"

# 1. 檢查並準備 uv 與 Python 3.12 虛擬環境
if ! command -v uv &> /dev/null; then
    echo "📦 未偵測到 uv，正在透過 Homebrew 或官方腳本安裝 uv..."
    if command -v brew &> /dev/null; then
        brew install uv
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

if [ ! -d "$ROOT_DIR/.venv" ]; then
    echo "🐍 建立 Python 3.12 虛擬環境 (.venv)..."
    uv venv "$ROOT_DIR/.venv" --python 3.12
fi

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
VENV_COMFY="$ROOT_DIR/.venv/bin/comfy"
UV_CACHE="$ROOT_DIR/.uv-cache"

echo "📦 安裝 comfy-cli、huggingface_hub、imageio-ffmpeg 與相關套件..."
uv pip install --python "$VENV_PYTHON" --cache-dir "$UV_CACHE" \
    comfy-cli \
    huggingface_hub \
    "huggingface_hub[hf_xet]" \
    tqdm \
    websocket-client \
    requests \
    imageio \
    imageio-ffmpeg \
    pillow \
    psutil \
    gguf

# 2. 建立本地 bin 目錄並連結 ffmpeg 與 comfy-cli
mkdir -p "$ROOT_DIR/bin" "$ROOT_DIR/src" "$ROOT_DIR/outputs"

FFMPEG_BIN=$("$VENV_PYTHON" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
if [ -f "$FFMPEG_BIN" ]; then
    ln -sf "$FFMPEG_BIN" "$ROOT_DIR/bin/ffmpeg"
    echo "✅ 已配置本地 ffmpeg -> $ROOT_DIR/bin/ffmpeg"
fi
ln -sf "$VENV_COMFY" "$ROOT_DIR/bin/comfy"
export PATH="$ROOT_DIR/bin:$ROOT_DIR/.venv/bin:$PATH"

# 3. 安裝或更新 ComfyUI
COMFY_DIR="$ROOT_DIR/ComfyUI"
if [ ! -d "$COMFY_DIR" ]; then
    echo "📥 正在下載 ComfyUI (最新版本，支援 MiniMax H3)..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
    echo "🔄 ComfyUI 目錄已存在，正在檢查更新..."
    git -C "$COMFY_DIR" pull --ff-only || true
fi

echo "📦 安裝 ComfyUI 核心依賴套件 (PyTorch MPS / Apple Silicon 支援)..."
export UV_HTTP_TIMEOUT=120
export UV_CONCURRENT_DOWNLOADS=4
uv pip install --python "$VENV_PYTHON" --cache-dir "$UV_CACHE" \
    torch torchvision torchaudio

# 過濾掉近 500MB 的非必要教學媒體包 (comfyui-workflow-templates)，僅保留 JSON 定義與核心依賴以大幅加速安裝
grep -v "^comfyui-workflow-templates" "$COMFY_DIR/requirements.txt" > "$ROOT_DIR/.comfy_reqs_fast.txt"
uv pip install --python "$VENV_PYTHON" --cache-dir "$UV_CACHE" \
    -r "$ROOT_DIR/.comfy_reqs_fast.txt" \
    comfyui-workflow-templates-core \
    comfyui-workflow-templates-json
rm -f "$ROOT_DIR/.comfy_reqs_fast.txt"

# 設定 comfy-cli 預設工作區 (先關閉 tracking 避免互動詢問)
echo "n" | "$VENV_COMFY" tracking disable 2>/dev/null || true
echo "n" | "$VENV_COMFY" --workspace="$COMFY_DIR" set-default "$COMFY_DIR" 2>/dev/null || true

# 4. 安裝 Custom Nodes
CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"

install_or_update_node() {
    local repo_url="$1"
    local dir_name="$2"
    local target_dir="$CUSTOM_NODES_DIR/$dir_name"
    if [ ! -d "$target_dir" ]; then
        echo "📥 安裝 Custom Node: $dir_name..."
        git clone "$repo_url" "$target_dir"
    else
        echo "🔄 更新 Custom Node: $dir_name..."
        git -C "$target_dir" pull --ff-only || true
    fi
    if [ -f "$target_dir/requirements.txt" ]; then
        echo "📦 安裝 $dir_name 依賴套件..."
        uv pip install --python "$VENV_PYTHON" --cache-dir "$UV_CACHE" -r "$target_dir/requirements.txt"
    fi
}

install_or_update_node "https://github.com/city96/ComfyUI-GGUF.git" "ComfyUI-GGUF"
install_or_update_node "https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash.git" "ComfyUI-HyperFlow-Flash"
install_or_update_node "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git" "seedvr2_videoupscaler"

# 套用 Apple Silicon M4 Max 無損加速補丁
"$VENV_PYTHON" "$ROOT_DIR/src/apply_mac_optimizations.py"

# 清理 uv 暫存快取，釋放磁碟空間
rm -rf "$UV_CACHE"

# 5. 建立模型目錄並下載模型 (支援斷點續傳)
mkdir -p \
    "$COMFY_DIR/models/diffusion_models" \
    "$COMFY_DIR/models/text_encoders" \
    "$COMFY_DIR/models/vae" \
    "$COMFY_DIR/models/SEEDVR2"

if [ "${SKIP_MODELS:-0}" = "1" ]; then
    echo "⏭️ 已設定 SKIP_MODELS=1，略過模型權重下載（將於執行 dev.sh / gen.sh 時自動檢查與補齊）。"
    "$VENV_PYTHON" "$ROOT_DIR/src/sulphur_assets.py" --comfy-dir "$COMFY_DIR" --skip-models
else
    echo "📥 正在檢查與下載 MiniMax H3 Ref2VA HyperFlow Flash (6步參考圖生片版) & SeedVR2 模型..."
    "$VENV_PYTHON" "$ROOT_DIR/src/download_models.py" --comfy-dir "$COMFY_DIR"
    echo "📥 正在檢查與下載 Sulphur 3 模型、H3 校正器 node 與 ComfyUI 工作流程..."
    "$VENV_PYTHON" "$ROOT_DIR/src/sulphur_assets.py" --comfy-dir "$COMFY_DIR"
fi

echo "========================================================"
echo "🎉 setup.sh 執行完畢！所有環境與工具已準備就緒。"
echo "👉 現在可以執行 ./dev.sh 或 ./gen.sh (Sulphur 3) 來產生影片！"
echo "========================================================"
