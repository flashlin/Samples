#!/bin/bash

# dk Docker Management Tool - Installation Script

set -e

echo "🐳 Installing dk - Docker Management Tool"
echo "=========================================="

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "✅ Rust installed successfully"
else
    echo "✅ Rust/Cargo found"
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
else
    echo "✅ Docker found"
fi

# Build the project
echo "🔨 Building dk..."
cargo build --release

# Install globally (optional)
read -p "📦 Install dk globally? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cargo install --path .
    echo "✅ dk installed globally. You can now use 'dk' from anywhere!"
else
    echo "✅ dk built successfully. Binary available at: ./target/release/dk"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Usage:"
echo "  dk              # Show container status"
echo "  dk logs         # Interactive log viewer"
echo "  dk bash         # Enter container shell"
echo "  dk rm           # Remove containers"
echo "  dk rm -f        # Force remove containers"
echo "  dk restart      # Restart containers"
echo ""
echo "For more information, run: dk --help"