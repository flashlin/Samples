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
echo "   Output Path: ../Generated"
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
rm -rf "../Generated"
mkdir -p "../Generated"
dotnet run -- "$swagger_url" -n "$sdk_name" -p "../Generated"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SDK generation completed successfully!"
    echo "📁 Generated project files in: Generated/"
    echo ""
    echo "📊 Generated directory contents:"
    ls -la "../Generated/" 2>/dev/null || echo "Generated directory not found"
    echo ""
    echo "🎉 Multi-target SDK project generated successfully!"
    echo "📦 Includes net8.0 and net9.0 builds with NuGet package!"
else
    echo ""
    echo "❌ SDK generation failed. Please check the errors above."
    exit 1
fi
