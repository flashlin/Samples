#!/bin/bash

# 當前目錄
PROJECT_DIR=$(pwd)

echo "🚀 Starting publish process for t1-web-components..."

# 1. 執行編譯
echo "📦 Building project..."
pnpm build

if [ $? -ne 0 ]; then
  echo "❌ Build failed. Aborting publish."
  exit 1
fi

# 2. 檢查 NPM 登入狀態 (選用)
echo "🔑 Checking NPM auth..."
npm whoami > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "⚠️ You are not logged in to NPM. Please run 'npm login' first."
  exit 1
fi

# 3. 發佈到 NPM
echo "🚀 Publishing to NPM..."
pnpm publish --access public

if [ $? -eq 0 ]; then
  echo "✅ Successfully published t1-web-components!"
else
  echo "❌ Publish failed."
  exit 1
fi
