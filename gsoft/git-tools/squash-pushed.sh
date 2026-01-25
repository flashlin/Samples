#!/bin/bash

# 檢查是否在 git 倉庫中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "錯誤：當前目錄不是 git 倉庫"
    exit 1
fi

# 獲取當前分支名稱
CURRENT_BRANCH=$(git branch --show-current)
echo "當前分支：$CURRENT_BRANCH"

# 檢查是否有遠端追蹤分支
if ! git rev-parse --abbrev-ref "$CURRENT_BRANCH"@{upstream} > /dev/null 2>&1; then
    echo "警告：當前分支沒有設定遠端追蹤分支"
    echo "請先設定遠端追蹤分支：git branch --set-upstream-to=origin/$CURRENT_BRANCH"
    exit 1
fi

# 自動獲取 commit 數量信息
TOTAL_COMMITS=$(git rev-list --count HEAD)

# 獲取與上游分支的差異 commits
UPSTREAM_COMMITS=0
if git rev-parse --abbrev-ref "$CURRENT_BRANCH"@{upstream} >/dev/null 2>&1; then
    UPSTREAM_COMMITS=$(git rev-list --count "$CURRENT_BRANCH"@{upstream}..HEAD 2>/dev/null || echo 0)
fi

# 獲取本地新 commits（未推送的）
LOCAL_COMMITS=$(git rev-list --count HEAD "@{u}"..HEAD 2>/dev/null || echo 0)

echo ""
echo "分支 commit 分析："
echo "總 commits：$TOTAL_COMMITS"
echo "與上游分支差異：$UPSTREAM_COMMITS commits"
echo "本地新 commits：$LOCAL_COMMITS commits"
echo ""

# 自動獲取最近的 commit 數量
if [ "$UPSTREAM_COMMITS" -gt 0 ]; then
    # 如果有未推送的 commits，預設值為未推送的數量
    DEFAULT_COUNT=$UPSTREAM_COMMITS
    echo "檢測到有 $UPSTREAM_COMMITS 個未推送的 commits"
elif [ "$LOCAL_COMMITS" -gt 0 ]; then
    # 如果有本地新 commits，預設值為本地 commits 數量
    DEFAULT_COUNT=$LOCAL_COMMITS
    echo "檢測到有 $LOCAL_COMMITS 個本地新 commits"
else
    # 如果沒有差異，使用總 commits 數量，但限制最大值
    if [ "$TOTAL_COMMITS" -gt 10 ]; then
        DEFAULT_COUNT=10
    else
        DEFAULT_COUNT=$TOTAL_COMMITS
    fi
fi

# 自動計算要壓縮的 commits 數量
# 使用實際 commits 數量-1（保留最舊的 commit 為 pick）
COMMIT_COUNT=$((TOTAL_COMMITS - 1))

# 如果 commits 數量不足 2，無法進行壓縮
if [ "$COMMIT_COUNT" -lt 1 ]; then
    echo "錯誤：分支上只有 1 個 commit，無法進行壓縮"
    exit 1
fi

echo "自動計算：將壓縮 $COMMIT_COUNT 個 commits（保留最舊的 1 個 commit）"

# 顯示最近的 commit 紀錄
echo ""
echo "最近的 $COMMIT_COUNT 個 commits："
git log --oneline -n "$COMMIT_COUNT"
echo ""

# 詢問新的 commit 訊息
read -p "請輸入新的 commit 訊息：" NEW_COMMIT_MESSAGE

if [ -z "$NEW_COMMIT_MESSAGE" ]; then
    echo "錯誤：commit 訊息不能為空"
    exit 1
fi

# 建立互動式 rebase 指令
REBASE_COMMAND="git rebase -i HEAD~$COMMIT_COUNT"

# 建立臨時檔案來處理 rebase
TEMP_FILE=$(mktemp)
cat > "$TEMP_FILE" << EOF
# 這是一個互動式 rebase 檔案
# 最舊的 commit 保持為 pick，其餘改為 squash
#
# 指令說明：
# p, pick = 使用 commit
# r, reword = 使用 commit，但修改 commit 訊息
# e, edit = 使用 commit，但暫停修改
# s, squash = 使用 commit，但合併到上一個 commit
# f, fixup = 類似 squash，但丟棄 commit 訊息
# d, drop = 移除 commit
#
# 這些行可以被重新排序，它們會從上到下執行
#
# 如果你移除某一行，對應的 commit 將會遺失
#
# 然而，如果你移除所有內容，rebase 將會中止
#
pick $(git log --format="%H" -n "$COMMIT_COUNT" | tail -1)
EOF

# 加入 squash 指令（除了第一個）
for ((i=COMMIT_COUNT-1; i>=1; i--)); do
    echo "squash $(git log --format="%H" -n "$COMMIT_COUNT" | tail -"$((i+1))" | head -1)" >> "$TEMP_FILE"
done

# 顯示 rebase 計畫
echo ""
echo "將執行的 rebase 計畫："
cat "$TEMP_FILE"
echo ""

# 詢問是否繼續
read -p "確定要執行 rebase 嗎？(y/N)：" CONFIRM_REBASE

if [[ "$CONFIRM_REBASE" != "y" && "$CONFIRM_REBASE" != "Y" ]]; then
    echo "取消操作"
    rm "$TEMP_FILE"
    exit 0
fi

# 執行 rebase
echo "正在執行 rebase..."
GIT_SEQUENCE_EDITOR="cat '$TEMP_FILE' >" git rebase -i HEAD~"$COMMIT_COUNT"

# 檢查 rebase 是否成功
if [ $? -eq 0 ]; then
    echo "rebase 成功完成"
    
    # 設定新的 commit 訊息
    git commit --amend -m "$NEW_COMMIT_MESSAGE"
    
    echo "commit 訊息已更新為：$NEW_COMMIT_MESSAGE"
    
    # 顯示新的 commit 紀錄
    echo ""
    echo "新的 commit 紀錄："
    git log --oneline -n 3
    echo ""
    
    # 詢問是否要 force push
    read -p "確定要執行 force push 到 origin/$CURRENT_BRANCH 嗎？(y/N)：" CONFIRM_PUSH
    
    if [[ "$CONFIRM_PUSH" == "y" || "$CONFIRM_PUSH" == "Y" ]]; then
        echo "正在執行 force push..."
        git push origin "$CURRENT_BRANCH" --force
        
        if [ $? -eq 0 ]; then
            echo "force push 成功完成"
        else
            echo "force push 失敗"
        fi
    else
        echo "取消 push 操作"
    fi
else
    echo "rebase 失敗，請手動處理衝突"
fi

# 清理臨時檔案
rm "$TEMP_FILE"