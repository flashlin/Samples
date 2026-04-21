#!/bin/bash

set -e

ALIAS_LINE='alias paste=/Users/flash/vdisk/github/Samples/gsoft/outputs/go-paste'

detect_shell_rc() {
    case "$(basename "$SHELL")" in
        zsh)  echo "$HOME/.zshrc" ;;
        bash) echo "$HOME/.bashrc" ;;
        *)    echo "$HOME/.profile" ;;
    esac
}

SHELL_RC=$(detect_shell_rc)

if grep -qF "$ALIAS_LINE" "$SHELL_RC" 2>/dev/null; then
    echo "Alias already exists in $SHELL_RC"
else
    echo "" >> "$SHELL_RC"
    echo "$ALIAS_LINE" >> "$SHELL_RC"
    echo "Added alias to $SHELL_RC"
    echo "Run 'source $SHELL_RC' or restart your terminal to use it"
fi
