#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/my-claude-code"
TARGET_DIR="$HOME/.claude"

show_usage() {
    echo "使用方式: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  --help       顯示此說明"
    echo ""
    echo "將 my-claude-code/ 目錄的所有內容複製到 ~/.claude/"
}

main() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "錯誤: 未知的選項 '$1'"
                show_usage
                exit 1
                ;;
        esac
    done

    if [ ! -d "$SOURCE_DIR" ]; then
        echo "錯誤: 來源目錄不存在: $SOURCE_DIR"
        exit 1
    fi

    echo "準備部署 Claude Code 設定..."
    echo "來源目錄: $SOURCE_DIR"
    echo "目標目錄: $TARGET_DIR"
    echo ""

    mkdir -p "$TARGET_DIR"

    for item in "$SOURCE_DIR"/*; do
        if [ -e "$item" ]; then
            item_name=$(basename "$item")
            echo "複製: $item_name"
            cp -r "$item" "$TARGET_DIR/"
        fi
    done

    echo ""
    echo "✓ 部署完成!"
    echo ""
    echo "已部署的內容:"
    ls -1 "$TARGET_DIR"
    echo ""
    echo "請重新啟動 Claude Code 以載入新的設定。"
}

main "$@"
