#!/bin/bash
set -e

EXT_NAME="lf-finder"
SRC_DIR="$HOME/vdisk/github/Samples/mac/Raycast/$EXT_NAME"

echo "🔧 開發 Raycast Extension: $EXT_NAME"

cd "$SRC_DIR"

# 安裝依賴
if [ ! -d "node_modules" ]; then
  echo "📦 安裝依賴套件..."
  pnpm install
fi

# 啟動開發模式
echo "🚀 啟動開發模式..."
echo "Raycast 會自動偵測到這個擴充功能"
pnpm run build

echo "✅ 開發模式已啟動，打開 Raycast 輸入 'Search Files' 測試"
