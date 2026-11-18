#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# dev.sh - 啟動 MiniMax H3 HyperFlow Flash (6步影音生成) + SeedVR2 放大管線
# 1. 若尚未安裝 ComfyUI 或虛擬環境，自動呼叫 ./setup.sh 初始化
# 2. 讀取本目錄中的 prompt.txt 提示詞檔案
# 3. 詢問使用者要產生的畫面大小 (預設 1024x768，更大則自動用 SeedVR2 放大) 與秒數 (預設 15 秒)
# 4. 顯示進度條，長影片自動以最後一格作為第一格接續生成，最終輸出至 outputs/
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d "$ROOT_DIR/.venv" ] || [ ! -d "$ROOT_DIR/ComfyUI" ]; then
    echo "⚙️ 偵測到環境尚未完整建立，先執行 ./setup.sh 準備環境..."
    bash "$ROOT_DIR/setup.sh"
fi

export PATH="$ROOT_DIR/bin:$ROOT_DIR/.venv/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1

exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/run_pipeline.py" "$@"
