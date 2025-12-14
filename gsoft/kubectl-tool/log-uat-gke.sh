#!/bin/bash

expandPath() {
    local path="$1"
    path="${path/#\~/$HOME}"
    echo "$path"
}

if ! command -v fzf &> /dev/null; then
    echo "錯誤: 找不到 fzf 指令"
    echo "請先安裝 fzf: brew install fzf"
    exit 1
fi

configPath=$(expandPath "~/Downloads/host-uat-gke.yaml")

echo "正在抓取 B2C namespace 的 pods..."
pods=$(kubectl get pods --kubeconfig="$configPath" -n=b2c -o jsonpath='{.items[*].metadata.name}')

if [ -z "$pods" ]; then
    echo "找不到任何 pods"
    exit 1
fi

podArray=($pods)
echo "找到 ${#podArray[@]} 個 pods"

selectedPod=$(printf '%s\n' "${podArray[@]}" | fzf --prompt="選擇 Pod > " --height=40% --layout=reverse --border)

if [ -z "$selectedPod" ]; then
    echo "未選擇任何 pod"
    exit 1
fi

echo ""
echo "=== Logs for pod: $selectedPod ==="
kubectl logs "$selectedPod" \
    --kubeconfig="$configPath" \
    -n=b2c \
    --tail=100
