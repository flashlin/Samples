#!/usr/bin/env bash

HF_ENV_FILE="${HF_ENV_FILE:-/Users/user/vdisk/qa-pair/my-tools/claude-code/.env}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_ENV_FILE" ]; then
    HF_TOKEN="$(grep -E '^HF_TOKEN=' "$HF_ENV_FILE" | tail -n 1 | cut -d= -f2- | tr -d "\"' \r")"
fi
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "🔑 HF_TOKEN loaded from $HF_ENV_FILE"
else
    unset HF_TOKEN
    echo "⚠️ HF_TOKEN not set in $HF_ENV_FILE, downloading anonymously (may be rate limited)"
fi
