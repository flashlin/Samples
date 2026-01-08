#!/bin/bash

expandPath() {
    local path="$1"
    path="${path/#\~/$HOME}"
    echo "$path"
}


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

echo "正在列出 b2c namespace 的 secrets..."
secrets=$(kubectl get secrets --kubeconfig="$configPath" -n=b2c -o jsonpath='{.items[*].metadata.name}')

if [ -z "$secrets" ]; then
    echo "找不到任何 secrets"
    exit 1
fi

secretArray=($secrets)
echo "找到 ${#secretArray[@]} 個 secrets"

selectedSecret=$(printf '%s\n' "${secretArray[@]}" | fzf --prompt="選擇 Secret > " --height=40% --layout=reverse --border)

if [ -z "$selectedSecret" ]; then
    echo "未選擇任何 secret"
    exit 1
fi

echo ""
echo "=== Secret: $selectedSecret ==="
kubectl get secret "$selectedSecret" -n b2c -o jsonpath='{.data}' \
    --kubeconfig="$configPath" \
    | jq 'map_values(@base64d)'
