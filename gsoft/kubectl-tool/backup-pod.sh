#!/bin/bash

# 取得腳本所在目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="$SCRIPT_DIR/pod-backups"

expandPath() {
    local path="$1"
    path="${path/#\~/$HOME}"
    echo "$path"
}

# 檢查 fzf 是否安裝
if ! command -v fzf &> /dev/null; then
    echo "錯誤: 找不到 fzf 指令"
    echo "請先安裝 fzf: brew install fzf"
    exit 1
fi

# 環境配置列表
configs=(
    "host-prod-gke.yaml"
    "host-staging-gke.yaml"
    "host-uat-gke.yaml"
    "rke2-prod-a.yaml"
    "rke2-prod-b.yaml"
    "rke2-stg.yaml"
)

# 選擇環境
selectedConfig=$(printf '%s\n' "${configs[@]}" | fzf --prompt="選擇環境 > " --height=40% --layout=reverse --border)

if [ -z "$selectedConfig" ]; then
    echo "未選擇任何環境"
    exit 1
fi

configPath=$(expandPath "~/Downloads/$selectedConfig")

if [ ! -f "$configPath" ]; then
    echo "錯誤: 找不到配置檔案 $configPath"
    exit 1
fi

echo "已選擇環境: $selectedConfig"

# 固定 namespace 為 b2c
selectedNamespace="b2c"
echo "使用 namespace: $selectedNamespace"

# 取得正在運作的 pods 和 image
echo "正在取得 $selectedNamespace namespace 的 running pods..."
podList=$(kubectl get pods -n="$selectedNamespace" --kubeconfig="$configPath" \
    --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}')

if [ -z "$podList" ]; then
    echo "找不到任何正在運作的 pods"
    exit 1
fi

# 計算 pod 數量
podCount=$(echo "$podList" | grep -c .)
echo "找到 $podCount 個正在運作的 pods"

# 用 fzf 選擇 pod (顯示 pod 名稱和 image)
selectedLine=$(echo "$podList" | fzf --prompt="選擇 Pod > " --height=40% --layout=reverse --border)

if [ -z "$selectedLine" ]; then
    echo "未選擇任何 pod"
    exit 1
fi

# 從選擇的行擷取 pod 名稱和 image
podName=$(echo "$selectedLine" | awk '{print $1}')
podImage=$(echo "$selectedLine" | awk '{print $2}')

echo ""
echo "=== 備份資訊 ==="
echo "Pod 名稱: $podName"
echo "Image 來源: $podImage"

# 建立備份目錄
mkdir -p "$BACKUP_DIR"

# 備份檔案路徑
backupFile="$BACKUP_DIR/${podName}.txt"

# 寫入備份檔案
echo "$podImage" > "$backupFile"

echo ""
echo "✅ 已成功備份到: $backupFile"
echo "備份內容: $podImage"
