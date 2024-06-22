start_marker="########## bash scripts lit"
end_marker="########## end"
bashrc_file="$HOME/.bashrc"

# 檢查 .bashrc 文件是否存在
if [ ! -f "$bashrc_file" ]; then
    echo ".bashrc 文件不存在"
    exit 1
fi

# 檢查開始標記是否存在
if grep -q "$start_marker" "$bashrc_file"; then
    # 創建一個臨時文件
    temp_file=$(mktemp)

    # 使用 sed 刪除指定區塊
    sed "/$start_marker/,/$end_marker/d" "$bashrc_file" > "$temp_file"

    # 將臨時文件的內容移回 .bashrc
    mv "$temp_file" "$bashrc_file"
    echo "移除舊的設定"
else
    echo "尚未安裝"
fi

cat setup-files/bash-script.sh >> "$bashrc_file"
