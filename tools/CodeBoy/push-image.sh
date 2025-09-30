#!/bin/bash
set -e

cd CodeBoyBackend
docker build -t codeboy-server:latest .
cd ..

cd CodeBoyFront
docker build -t codeboy-front:latest .
cd ..


if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    exit 1
fi

source .env

if [ -z "$DockerRegistryServer" ]; then
    echo "❌ DockerRegistryServer not set in .env file!"
    exit 1
fi

echo "🏷️  Tagging and pushing Docker images..."
echo "📦 Registry: $DockerRegistryServer"
echo ""

echo "🔖 Tagging codeboy-server:latest..."
docker tag codeboy-server:latest ${DockerRegistryServer}codeboy-server:latest

echo "📤 Pushing codeboy-server:latest..."
docker push ${DockerRegistryServer}codeboy-server:latest

echo ""
echo "🔖 Tagging codeboy-front:latest..."
docker tag codeboy-front:latest ${DockerRegistryServer}codeboy-front:latest

echo "📤 Pushing codeboy-front:latest..."
docker push ${DockerRegistryServer}codeboy-front:latest

echo ""
echo "✅ All images tagged and pushed successfully!"
