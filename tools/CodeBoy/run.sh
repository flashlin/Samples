#!/bin/bash

# This script generates SDK from Swagger URL

echo "🚀 CodeGen Tool"
echo "================================================"

# Default values
default_swagger_url="https://steropes-api.sbotry.com/swagger/v1/swagger.json"
default_sdk_name="Steropes"

# Ask for Swagger URL with default value
echo -n "Input swagger url (default: $default_swagger_url): "
read swagger_url

# Use default if empty
if [ -z "$swagger_url" ]; then
    swagger_url="$default_swagger_url"
    echo "Using default: $swagger_url"
fi

# Ask for SDK name with default value
echo -n "SDK name (default: $default_sdk_name): "
read sdk_name

# Use default if empty
if [ -z "$sdk_name" ]; then
    sdk_name="$default_sdk_name"
    echo "Using default: $sdk_name"
fi

echo ""
echo "📋 Configuration:"
echo "   Swagger URL: $swagger_url"
echo "   SDK Name: $sdk_name"
echo "   Output: ../${sdk_name}Client.cs"
echo ""

# Navigate to the project directory
cd CodeGen

# Build the project first
echo "📦 Building project..."
dotnet build

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Execute the tool with user inputs
echo "🔍 Generating SDK from Swagger..."
mkdir -p "../Generated"
dotnet run -- "$swagger_url" -n "$sdk_name" -o "../Generated/${sdk_name}Client.cs"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SDK generation completed successfully!"
    echo "📄 Generated file: Generated/${sdk_name}Client.cs"
    echo ""
    echo "📊 File size:"
    ls -lh "../Generated/${sdk_name}Client.cs" 2>/dev/null || echo "Output file not found"
    echo ""
    echo "🎉 You can now use the generated Generated/${sdk_name}Client.cs in your project!"
else
    echo ""
    echo "❌ SDK generation failed. Please check the errors above."
    exit 1
fi
