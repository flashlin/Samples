#!/bin/bash

# CodeBoy Server Docker Management Script
# Interactive management of CodeBoy Server Docker container using fzf

set -e

echo "🐳 CodeBoy Server Docker Management"
echo ""

# Use fzf to select action
action=$(echo -e "start\nstop" | fzf --prompt="Select action: " --height=40% --border --header="Choose what to do with CodeBoy Server")

# Check if user cancelled selection
if [ -z "$action" ]; then
    echo "❌ No action selected. Exiting..."
    exit 0
fi

case $action in
    "start")
        echo "🚀 Starting CodeBoy Server..."
        
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
        ;;
        
    "stop")
        echo "🛑 Stopping CodeBoy Server..."
        
        # Check if container is running
        if docker ps | grep -q codeboy-server; then
            docker stop codeboy-server
            docker rm codeboy-server
            echo "✅ CodeBoy Server stopped and removed successfully!"
        else
            echo "ℹ️  CodeBoy Server is not running."
        fi
        ;;
        
    *)
        echo "❌ Invalid action: $action"
        exit 1
        ;;
esac
