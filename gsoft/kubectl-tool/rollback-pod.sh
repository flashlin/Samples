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
currentImage=$(echo "$selectedLine" | awk '{print $2}')

echo ""
echo "=== 目前運行資訊 ==="
echo "Pod 名稱: $podName"
echo "目前 Image: $currentImage"

# 檢查備份檔案是否存在
serviceName=$(echo "$podName" | sed 's/-[^-]*-[^-]*$//')
backupFile="$BACKUP_DIR/${serviceName}.txt"

if [ ! -f "$backupFile" ]; then
    echo ""
    echo "❌ 錯誤: 找不到備份檔案 $backupFile"
    echo "請先使用 backup-pod.sh 備份此 pod 的 image"
    exit 1
fi

# 讀取備份的 image
backupImage=$(cat "$backupFile")

echo ""
echo "=== 備份檔案資訊 ==="
echo "備份檔案: $backupFile"
echo "備份 Image: $backupImage"

# 比較目前和備份的 image
if [ "$currentImage" == "$backupImage" ]; then
    echo ""
    echo "ℹ️  目前運行的 image 與備份相同，無需還原"
    exit 0
fi

echo ""
echo "=== 差異比較 ==="
echo "目前 Image: $currentImage"
echo "備份 Image: $backupImage"

# 詢問是否還原
echo ""
read -p "是否要用備份檔案中的 image 還原? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消還原操作"
    exit 0
fi

# 取得 pod 的 owner (ReplicaSet)，然後推斷 deployment 名稱
echo ""
echo "正在取得 deployment 資訊..."

# 從 pod 取得 ReplicaSet 名稱
replicaSetName=$(kubectl get pod "$podName" -n="$selectedNamespace" --kubeconfig="$configPath" \
    -o jsonpath='{.metadata.ownerReferences[?(@.kind=="ReplicaSet")].name}')

if [ -z "$replicaSetName" ]; then
    echo "❌ 錯誤: 無法取得 ReplicaSet 名稱，此 pod 可能不是由 Deployment 管理"
    exit 1
fi

# 從 ReplicaSet 取得 Deployment 名稱
deploymentName=$(kubectl get replicaset "$replicaSetName" -n="$selectedNamespace" --kubeconfig="$configPath" \
    -o jsonpath='{.metadata.ownerReferences[?(@.kind=="Deployment")].name}')

if [ -z "$deploymentName" ]; then
    echo "❌ 錯誤: 無法取得 Deployment 名稱"
    exit 1
fi

# 取得容器名稱
containerName=$(kubectl get pod "$podName" -n="$selectedNamespace" --kubeconfig="$configPath" \
    -o jsonpath='{.spec.containers[0].name}')

echo "Deployment: $deploymentName"
echo "Container: $containerName"

# 執行 image 更新
echo ""
echo "正在更新 deployment/$deploymentName 的 image..."
kubectl set image deployment/"$deploymentName" \
    "$containerName=$backupImage" \
    -n="$selectedNamespace" \
    --kubeconfig="$configPath"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 還原成功!"
    echo "已將 $deploymentName 的 image 更新為: $backupImage"
    echo ""
    echo "可使用以下指令查看 rollout 狀態:"
    echo "kubectl rollout status deployment/$deploymentName -n=$selectedNamespace --kubeconfig=\"$configPath\""
else
    echo ""
    echo "❌ 還原失敗，請檢查錯誤訊息"
    exit 1
fi
