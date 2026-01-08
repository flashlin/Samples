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
echo "正在列出 secret '$selectedSecret' 的 keys..."
existingKeys=$(kubectl get secret "$selectedSecret" -n b2c -o jsonpath='{.data}' \
    --kubeconfig="$configPath" | jq -r 'keys[]')

if [ -z "$existingKeys" ]; then
    echo "該 secret 沒有任何 keys"
    exit 1
fi

delKey=$(printf '%s\n' $existingKeys | fzf --prompt="選擇要刪除的 Key > " --height=40% --layout=reverse --border)

if [ -z "$delKey" ]; then
    echo "未選擇任何 key"
    exit 1
fi

# 確認刪除
echo ""
echo "=========================================="
echo "確認刪除以下資訊："
echo "  環境:      $selectedConfig"
echo "  Secret:    $selectedSecret"
echo "  Key:       $delKey"
echo "=========================================="
echo ""
read -p "確定要刪除嗎？請輸入 yes 確認: " confirm

if [ "$confirm" != "yes" ]; then
    echo "已取消刪除"
    exit 0
fi

# 刪除 key
echo "正在刪除 key '$delKey'..."
kubectl patch secret "$selectedSecret" -n b2c \
    --kubeconfig="$configPath" \
    --type='json' \
    -p="[{\"op\": \"remove\", \"path\": \"/data/$delKey\"}]"

if [ $? -eq 0 ]; then
    echo "成功刪除 key '$delKey' from secret '$selectedSecret'"
else
    echo "刪除失敗"
    exit 1
fi
