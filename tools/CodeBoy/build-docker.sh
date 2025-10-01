#!/bin/bash

# CodeBoy Server Docker Build Script

set -e

echo "🐳 Building CodeBoy Server Docker Image..."

# Change to CodeBoyServer directory and build the Docker image
cd CodeBoyBackend
docker build -t codeboy-server:latest .
cd ..

cd CodeBoyFront
docker build -t codeboy-front:latest .
cd ..

echo "✅ Docker image built successfully!"

