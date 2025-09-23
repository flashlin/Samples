#!/bin/bash

# CodeBoy Server Docker Run Script

set -e

echo "🚀 Starting CodeBoy Server with Docker..."

# Check if image exists
if ! docker images codeboy-server:latest | grep -q codeboy-server; then
    echo "❌ Docker image not found. Building it first..."
    ./build-docker.sh
fi

echo "🐳 Starting container..."

# Stop and remove existing container if running
docker stop codeboy-server 2>/dev/null || true
docker rm codeboy-server 2>/dev/null || true

# Run the container
docker run -d \
    --name codeboy-server \
    -p 8080:8080 \
    --restart unless-stopped \
    codeboy-server:latest

echo "✅ CodeBoy Server is running!"
echo ""
echo "📖 API Documentation: http://localhost:8080"
echo "🔍 Health Check: http://localhost:8080/health"
echo "📊 Container Status:"

# Show container status
docker ps | grep codeboy-server

echo ""
echo "📝 To view logs:"
echo "  docker logs -f codeboy-server"
echo ""
echo "🛑 To stop:"
echo "  docker stop codeboy-server"
