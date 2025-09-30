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

        # Build images using docker compose
        docker compose -f docker-compose-local.yml build
        # Run the container
        docker compose -f docker-compose-local.yml up -d

        echo "✅ CodeBoy Server is running!"
        echo ""
        echo "📖 API Documentation: http://localhost:8080"
        echo "🔍 Health Check: http://localhost:8080/health"
        echo "📊 Container Status:"
        ;;
        
    "stop")
        echo "🛑 Stopping CodeBoy Server..."
        
        # Check if container is running
        if docker ps | grep -q codeboy-server; then
            docker compose -f docker-compose-local.yml down
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
