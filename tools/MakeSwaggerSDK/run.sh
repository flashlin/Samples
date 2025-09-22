#!/bin/bash

# MakeSwaggerSDK Runner Script
# This script demonstrates how to use the MakeSwaggerSDK tool with Petstore API

echo "🚀 Running MakeSwaggerSDK with Petstore API..."
echo "================================================"

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

# Run the MakeSwaggerSDK tool with Petstore API
echo "🔍 Parsing Swagger from Petstore API..."
echo "URL: https://petstore.swagger.io/v2/swagger.json"
echo ""

# Execute the tool with PetStore API
dotnet run -- "https://petstore.swagger.io/v2/swagger.json" -n PetStore -o PetStoreClient.cs

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SDK generation completed successfully!"
    echo "📄 Generated file: PetStoreClient.cs"
    echo ""
    echo "📊 File size:"
    ls -lh PetStoreClient.cs 2>/dev/null || echo "Output file not found"
    echo ""
    echo "🎉 You can now use the generated PetStoreClient.cs in your project!"
else
    echo ""
    echo "❌ SDK generation failed. Please check the errors above."
    exit 1
fi
