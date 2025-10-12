#!/bin/bash
set -e

echo "🧹 Cleaning TsSql..."
rm -rf dist
rm -rf node_modules/.vite

echo "📦 Installing dependencies..."
pnpm install

echo "🔨 Building TypeScript definitions..."
pnpm exec tsc

echo "🔨 Building with Vite..."
pnpm exec vite build

echo "✅ Build complete!"
echo ""
echo "📁 Generated files in dist/:"
ls -la dist/ | head -15
echo ""
echo "🔍 Verifying type definitions..."
if [ -f "dist/index.d.ts" ]; then
  echo "✅ dist/index.d.ts exists"
  echo "📝 Content:"
  cat dist/index.d.ts
else
  echo "❌ dist/index.d.ts missing!"
  exit 1
fi

