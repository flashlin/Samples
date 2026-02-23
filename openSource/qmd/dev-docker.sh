#!/bin/bash
set -e

# ==========================================
# qmd MCP Server Docker Runner
# ==========================================

# 1. 確保宿主機上所需的目錄存在
mkdir -p ./documents

# 2. 安靜地建立 Docker 映像檔
# 將標準輸出重導向至 stderr，以避免污染 MCP 的 JSON-RPC stdout
>&2 echo "=== 正在建立 qmd-mcp Docker 映像檔 ==="
docker build -q -t qmd-mcp . >&2
>&2 echo "=== 建立完成。啟動 qmd MCP server ==="

# 3. 清理可能正在運行的舊容器，避免 port 衝突
OLD_CONTAINERS=$(docker ps -a -q -f ancestor=qmd-mcp)
if [ -n "$OLD_CONTAINERS" ]; then
  >&2 echo "=== 清理舊的 qmd-mcp 容器 ==="
  docker rm -f $OLD_CONTAINERS >&2
fi

# 4. 啟動容器
# - -i 保持標準輸入開啟 (請勿加上 -t，否則會破壞 MCP 的 JSON-RPC 通訊)
# - 映射宿主的 ./documents 到容器內的輸入文件目錄
# - 映射宿主的 ~/.cache/qmd 到容器內的 cache 目錄（共享 index.sqlite 和 models）
docker run -i --rm \
  -p 8181:8181 \
  -v "$(pwd)/documents:/documents" \
  -v "$HOME/.cache/qmd:/root/.cache/qmd" \
  qmd-mcp mcp "$@"
