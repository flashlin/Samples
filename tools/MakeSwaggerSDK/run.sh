#!/bin/bash

# MakeSwaggerSDK Runner Script
# This script generates SDK from Swagger URL

echo "🚀 MakeSwaggerSDK Tool"
echo "================================================"

# Ask for Swagger URL
echo -n "Input swagger url: "
read swagger_url

# Check if swagger_url is empty
if [ -z "$swagger_url" ]; then
    echo "❌ Swagger URL cannot be empty. Exiting..."
    exit 1
fi

# Ask for SDK name
echo -n "SDK name: "
read sdk_name

# Check if sdk_name is empty
if [ -z "$sdk_name" ]; then
    echo "❌ SDK name cannot be empty. Exiting..."
    exit 1
fi

echo ""
echo "📋 Configuration:"
echo "   Swagger URL: $swagger_url"
echo "   SDK Name: $sdk_name"
echo "   Output: ../${sdk_name}Client.cs"
echo ""

# Navigate to the project directory
cd MakeSwaggerSDK

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
dotnet run -- "$swagger_url" -n "$sdk_name" -o "../${sdk_name}Client.cs"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SDK generation completed successfully!"
    echo "📄 Generated file: ${sdk_name}Client.cs"
    echo ""
    echo "📊 File size:"
    ls -lh "../${sdk_name}Client.cs" 2>/dev/null || echo "Output file not found"
    echo ""
    echo "🎉 You can now use the generated ${sdk_name}Client.cs in your project!"
else
    echo ""
    echo "❌ SDK generation failed. Please check the errors above."
    exit 1
fi
