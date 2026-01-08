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
read -p "請輸入要新增的 key: " newKey

if [ -z "$newKey" ]; then
    echo "未輸入 key"
    exit 1
fi

read -p "請輸入要新增的 value: " newValue

if [ -z "$newValue" ]; then
    echo "未輸入 value"
    exit 1
fi

# 檢查 key 是否已存在
existingKeys=$(kubectl get secret "$selectedSecret" -n b2c -o jsonpath='{.data}' \
    --kubeconfig="$configPath" | jq -r 'keys[]')

for key in $existingKeys; do
    if [ "$key" == "$newKey" ]; then
        echo "錯誤: key '$newKey' already exists"
        exit 1
    fi
done

# 新增 key/value 到 secret
echo "正在新增 key '$newKey' 到 secret '$selectedSecret'..."
kubectl patch secret "$selectedSecret" -n b2c \
    --kubeconfig="$configPath" \
    --type='json' \
    -p="[{\"op\": \"add\", \"path\": \"/data/$newKey\", \"value\": \"$(echo -n "$newValue" | base64)\"}]"

if [ $? -eq 0 ]; then
    echo "成功新增 key '$newKey' 到 secret '$selectedSecret'"
else
    echo "新增失敗"
    exit 1
fi
