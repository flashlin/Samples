#!/bin/bash
set -e

echo "🔨 Building TsSql library..."
cd ../TsSql
pnpm install
pnpm run build

echo "🔨 Building VimComponent..."
cd ../VimComponent
pnpm install
pnpm run build

echo "🔨 Building VimDemo..."
cd ../VimDemo
echo "🧹 Cleaning pnpm cache for local dependencies..."
rm -rf node_modules/.pnpm/t1-vim-editor*
rm -rf node_modules/.pnpm/t1-tssql*
rm -rf node_modules/t1-vim-editor
rm -rf node_modules/t1-tssql
rm -rf node_modules/.vite
rm -rf dist
pnpm install

echo "🚀 Starting dev server..."
pnpm run dev