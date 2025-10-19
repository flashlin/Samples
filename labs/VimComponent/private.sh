#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="http://localhost:4873"

echo "=========================================="
echo "Publishing packages to private registry"
echo "Registry: $REGISTRY"
echo "=========================================="
echo ""

get_registry_version() {
    local package_name=$1
    local registry=$2
    
    local version=$(npm view "$package_name" version --registry="$registry" 2>/dev/null || echo "")
    echo "$version"
}

get_local_version() {
    local package_dir=$1
    
    local version=$(node -p "require('./$package_dir/package.json').version" 2>/dev/null || echo "")
    echo "$version"
}

publish_package() {
    local package_dir=$1
    local package_name=$2
    
    echo "Publishing $package_name..."
    echo "------------------------------------------"
    
    local registry_version=$(get_registry_version "$package_name" "$REGISTRY")
    local local_version=$(get_local_version "$package_dir")
    
    echo "Local version: $local_version"
    echo "Registry version: $registry_version"
    
    if [ "$registry_version" = "$local_version" ] && [ -n "$registry_version" ]; then
        echo "⏭️  Skipping $package_name - version $local_version already published"
        echo ""
        return 0
    fi
    
    cd "$SCRIPT_DIR/$package_dir"
    
    echo "Building $package_name..."
    npm run build
    
    if [ $? -ne 0 ]; then
        echo "❌ Build failed for $package_name"
        return 1
    fi
    
    echo "Publishing $package_name to $REGISTRY..."
    npm run publish:private
    
    if [ $? -ne 0 ]; then
        echo "❌ Publish failed for $package_name"
        return 1
    fi
    
    echo "✅ Successfully published $package_name"
    echo ""
}

publish_package "TsSql" "@mrbrain/t1-tssql"
publish_package "VimComponent" "@mrbrain/t1-vim-editor"

echo "=========================================="
echo "✅ All packages published successfully!"
echo "=========================================="

