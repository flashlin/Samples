#!/bin/bash

# VimComponent 更新並啟動 VimDemo 腳本
# Usage: ./update-and-run.sh

set -e  # Exit on error

echo "🔨 Building VimComponent..."
cd "$(dirname "$0")/VimComponent"
pnpm run build

echo ""
echo "📦 Updating VimDemo dependencies..."
cd ../VimDemo
pnpm install

echo ""
echo "✅ Update complete!"
echo ""
echo "🚀 Starting development server..."
echo "   Press Ctrl+C to stop"
echo ""
pnpm run dev

