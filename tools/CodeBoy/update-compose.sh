#!/bin/bash

set -e

if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    exit 1
fi

source .env

if [ -z "$DockerRegistryServer" ]; then
    echo "❌ DockerRegistryServer not set in .env file!"
    exit 1
fi

if [ ! -f docker-compose-temp.yml ]; then
    echo "❌ docker-compose-temp.yml not found!"
    exit 1
fi

echo "🔧 Updating docker-compose configuration..."
echo "📦 Registry: $DockerRegistryServer"
echo ""

sed "s|{{DOCKER_REGISTRY}}|${DockerRegistryServer}|g" docker-compose-temp.yml > docker-compose.yml

echo "✅ docker-compose.yml updated successfully!"
echo ""
echo "📄 Updated services:"
echo "  - codeboy-server: ${DockerRegistryServer}codeboy-server:latest"
echo "  - codeboy-front: ${DockerRegistryServer}codeboy-front:latest"
