#!/bin/bash

expandPath() {
    local path="$1"
    path="${path/#\~/$HOME}"
    echo "$path"
}

if ! command -v jq &> /dev/null; then
    echo "錯誤: 找不到 jq 指令"
    echo "請先安裝 jq: brew install jq"
    exit 1
fi

if ! command -v fzf &> /dev/null; then
    echo "錯誤: 找不到 fzf 指令"
    echo "請先安裝 fzf: brew install fzf"
    exit 1
fi

configs=(
    "host-prod-gke.yaml"
    "host-staging-gke.yaml"
    "host-uat-gke.yaml"
    "rke2-prod-a.yaml"
    "rke2-prod-b.yaml"
    "rke2-stg.yaml"
)

selectedConfig=$(printf '%s\n' "${configs[@]}" | fzf --prompt="選擇環境 > " --height=40% --layout=reverse --border)

if [ -z "$selectedConfig" ]; then
    echo "未選擇任何環境"
    exit 1
fi

configPath=$(expandPath "~/Downloads/$selectedConfig")

echo "正在讀取 akis-secret..."
kubectl get secret akis-secret -n b2c -o jsonpath='{.data}' \
    --kubeconfig="$configPath" \
    | jq 'map_values(@base64d)'
