#!/bin/bash

# CodeBoy Server Docker Build Script

set -e

echo "🐳 Building CodeBoy Server Docker Image..."

# Change to CodeBoyServer directory and build the Docker image
cd CodeBoyBackend
docker build -t codeboy-server:latest .

echo "✅ Docker image built successfully!"

# Show the image
docker images codeboy-server:latest

echo ""
echo "🚀 To run the container:"
echo "  docker run -p 8080:8080 codeboy-server:latest"
echo ""
echo "🐙 Or using docker-compose:"
echo "  docker-compose up -d"
echo ""
echo "📖 API Documentation available at:"
echo "  http://localhost:8080"
echo ""
echo "🔍 Health check endpoint:"
echo "  http://localhost:8080/health"
