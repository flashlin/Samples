#!/bin/bash

# 檢查是否在 git 專案中
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "錯誤: 這不是一個 git 專案。"
    exit 1
fi

# 取得最近的 commit 數量 (最多 20)
TOTAL_COMMITS=$(git rev-list --count HEAD)
COUNT=$(( TOTAL_COMMITS < 20 ? TOTAL_COMMITS : 20 ))

# 取得 commit 清單 (由新到舊)
# 格式: hash subject
COMMITS_LIST=$(git log -n "$COUNT" --format="%h %s")

# 使用 fzf 選取要 squash 的 commits
# --multi 允許空白鍵多選
SELECTED=$(echo "$COMMITS_LIST" | fzf --multi \
    --header "按 [空白鍵] 選取要 Squash 的 Commit (合併至上方)，[Enter] 開始 Rebase" \
    --preview "git show --color=always {1}" \
    --preview-window=right:60%)

# 如果使用者取消 (Esc) 則離開
if [ $? -ne 0 ] || [ -z "$SELECTED" ]; then
    echo "未選取或已取消。"
    exit 0
fi

# 提取被選取的 hashes
SELECTED_HASHES=$(echo "$SELECTED" | awk '{print $1}')

# 產生暫存的 rebase todo 清單
# 注意：我們保持「由新到舊」的順序，因為 git rebase 會按檔案順序執行
TODO_FILE=$(mktemp)
FIRST=true

while read -r line; do
    [ -z "$line" ] && continue
    HASH=$(echo "$line" | awk '{print $1}')
    SUBJECT=$(echo "$line" | cut -d' ' -f2-)
    
    # 檢查此 HASH 是否在選取名單中
    IS_SELECTED=false
    if echo "$SELECTED_HASHES" | grep -q "$HASH"; then
        IS_SELECTED=true
    fi

    if $FIRST; then
        # 第一個 (最上方/最新) 必須是 pick
        echo "pick $HASH $SUBJECT" >> "$TODO_FILE"
        FIRST=false
    elif [ "$IS_SELECTED" = true ]; then
        echo "squash $HASH $SUBJECT" >> "$TODO_FILE"
    else
        echo "pick $HASH $SUBJECT" >> "$TODO_FILE"
    fi
done <<< "$COMMITS_LIST"

# 執行 Rebase
# 由於我們是由新到舊排列，我們需要告訴 git rebase 不要重新排序
# 通常 git rebase todo 是舊到新，但這裡我們強制餵入由新到舊的清單
# git 會按照 todo 檔案裡的順序從第一行開始 apply
echo "正在啟動互動式 Rebase..."

if [ "$TOTAL_COMMITS" -le "$COUNT" ]; then
    GIT_SEQUENCE_EDITOR="cat $TODO_FILE >" git rebase -i --root
else
    # 取得第 N 個 commit 的父節點作為 base
    BASE_HASH=$(git rev-parse "HEAD~$COUNT")
    GIT_SEQUENCE_EDITOR="cat $TODO_FILE >" git rebase -i "$BASE_HASH"
fi

rm "$TODO_FILE"
