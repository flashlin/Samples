#!/bin/bash
set -e

echo "🔨 Building TsSql library..."
cd ../TsSql
pnpm install
pnpm run build

echo "🔨 Building VimComponent..."
cd ../VimComponent
pnpm run build

echo "🔨 Building VimDemo..."
cd ../VimDemo
rm -rf node_modules/vimcomponent
rm -rf node_modules/tssql
pnpm install

echo "🚀 Starting dev server..."
pnpm run dev