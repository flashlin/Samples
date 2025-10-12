#!/bin/bash

# TsSql Setup Script

echo "🚀 Setting up TsSql project..."

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null
then
    echo "❌ pnpm is not installed. Please install it first:"
    echo "   npm install -g pnpm"
    exit 1
fi

# Check Node version
if [ -f ".nvmrc" ]; then
    REQUIRED_NODE_VERSION=$(cat .nvmrc)
    echo "📌 Required Node.js version: $REQUIRED_NODE_VERSION"
    
    if command -v nvm &> /dev/null
    then
        echo "Using nvm to switch to Node.js $REQUIRED_NODE_VERSION..."
        nvm use
    else
        echo "⚠️  nvm not found. Please make sure you're using Node.js $REQUIRED_NODE_VERSION"
    fi
fi

# Install dependencies
echo "📦 Installing dependencies..."
pnpm install

if [ $? -eq 0 ]; then
    echo "✅ Setup completed successfully!"
    echo ""
    echo "Available commands:"
    echo "  pnpm run dev      - Start development server"
    echo "  pnpm run build    - Build for production"
    echo "  pnpm test         - Run tests"
    echo "  pnpm run test:ui  - Run tests with UI"
    echo ""
    echo "🎉 You can now run: pnpm run dev"
else
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi

