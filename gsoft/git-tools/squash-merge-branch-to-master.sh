#!/bin/bash

# 列出所有本地分支，使用 fzf 讓使用者選擇
echo "請選擇要合併的分支："
selected_branch=$(git branch --format='%(refname:short)' | fzf --height=40% --reverse --prompt="選擇分支 > ")

# 檢查是否有選擇分支
if [ -z "$selected_branch" ]; then
    echo "未選擇任何分支，離開。"
    exit 1
fi

echo "你選擇了分支: $selected_branch"

# 切換到 master 分支
git checkout master

# 執行 squash merge
git merge --squash "$selected_branch"

# 詢問使用者輸入 commit message
echo ""
echo "請輸入 commit message（留空則取消）："
read -r commit_message

# 檢查是否有輸入 commit message
if [ -z "$commit_message" ]; then
    echo "未輸入 commit message，取消合併。"
    git reset --hard HEAD
    exit 1
fi

# 執行 commit
git commit -m "$commit_message"

echo ""
echo "合併完成，請檢查合併結果並手動推送到遠端 master"
