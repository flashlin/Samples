#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="https://registry.npmjs.org"

echo "=========================================="
echo "Publishing packages to public registry"
echo "Registry: $REGISTRY"
echo "=========================================="
echo ""

publish_package() {
    local package_dir=$1
    local package_name=$2
    
    echo "Publishing $package_name..."
    echo "------------------------------------------"
    
    cd "$SCRIPT_DIR/$package_dir"
    
    echo "Building $package_name..."
    npm run build
    
    if [ $? -ne 0 ]; then
        echo "❌ Build failed for $package_name"
        return 1
    fi
    
    echo "Publishing $package_name to $REGISTRY..."
    npm run publish:public
    
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

