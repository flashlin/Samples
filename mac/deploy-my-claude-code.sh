#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/claude-code-skills"
DEFAULT_TARGET_DIR="$HOME/.claude/skills"
PROJECT_TARGET_DIR="$SCRIPT_DIR/.claude/skills"

show_usage() {
    echo "使用方式: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  --project    部署到專案目錄 (.claude/skills/)"
    echo "  --help       顯示此說明"
    echo ""
    echo "預設會部署到個人 skills 目錄: ~/.claude/skills/"
}

main() {
    TARGET_DIR="$DEFAULT_TARGET_DIR"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --project)
                TARGET_DIR="$PROJECT_TARGET_DIR"
                shift
                ;;
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

    echo "準備部署 Claude Code Skills..."
    echo "來源目錄: $SOURCE_DIR"
    echo "目標目錄: $TARGET_DIR"
    echo ""

    mkdir -p "$TARGET_DIR"

    for skill_dir in "$SOURCE_DIR"/*; do
        if [ -d "$skill_dir" ]; then
            skill_name=$(basename "$skill_dir")
            echo "複製 skill: $skill_name"
            cp -r "$skill_dir" "$TARGET_DIR/"
        fi
    done

    echo ""
    echo "✓ 部署完成!"
    echo ""
    echo "已部署的 skills:"
    ls -1 "$TARGET_DIR"
    echo ""
    echo "請重新啟動 Claude Code 以載入新的 skills。"
}

main "$@"
