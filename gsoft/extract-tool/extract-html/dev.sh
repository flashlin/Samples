#!/usr/bin/env bash
cd "$(dirname "$0")"
uv run python -m extract_html.app
