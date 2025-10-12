#!/bin/bash
set -e

echo "🛑 請先停止當前的 dev server (Ctrl+C)"
echo "按 Enter 繼續..."
read

echo ""
echo "🔨 Step 1: Building VimComponent..."
cd ../VimComponent
pnpm run build
echo "✅ VimComponent built"
echo ""

echo "🧹 Step 2: Cleaning ALL caches..."
cd ../VimDemo
rm -rf node_modules/.pnpm/vimcomponent*
rm -rf node_modules/vimcomponent
rm -rf node_modules/.vite
rm -rf dist
echo "✅ All caches cleaned"
echo ""

echo "📦 Step 3: Reinstalling dependencies..."
pnpm install
echo "✅ Dependencies installed"
echo ""

echo "🔍 Step 4: Verifying installation..."
INSTALLED_FILE=$(find node_modules/.pnpm -path "*/vimcomponent/dist/vim-editor.es.js" 2>/dev/null | head -1)
if [ -n "$INSTALLED_FILE" ]; then
  echo "✅ Found: $INSTALLED_FILE"
  if grep -q "DEBUG.*Key pressed" "$INSTALLED_FILE"; then
    echo "✅ File contains DEBUG log code"
  else
    echo "❌ File does NOT contain DEBUG log code!"
    exit 1
  fi
else
  echo "❌ VimComponent not found!"
  exit 1
fi
echo ""

echo "🚀 Step 5: Starting dev server..."
echo ""
echo "📝 After the server starts:"
echo "   1. Open browser to http://localhost:5173"
echo "   2. Press Cmd+Shift+R (hard refresh)"
echo "   3. Press F12 (open Console)"
echo "   4. Click on vim-editor"
echo "   5. Press ANY key (like 'h')"
echo "   6. You should see [DEBUG] log"
echo ""
echo "Starting in 3 seconds..."
sleep 3

pnpm run dev

