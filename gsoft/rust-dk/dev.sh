#!/bin/bash

echo "🔨 Building dk tool..."

cargo build --release

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo "📦 Copying binary to parent directory..."
    cp target/release/dk ../
    echo "✅ Binary copied to ../dk"
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🚀 Running dk tool..."
echo "Available commands:"
echo "  dk          - Show container status"
echo "  dk logs     - View container logs"
echo "  dk bash     - Enter container shell"
echo "  dk rm       - Remove containers"
echo "  dk rm -f    - Force remove containers"
echo "  dk restart  - Restart containers"
echo "  dk clean    - Clean up Docker images and cache"
echo ""

cargo run --release
