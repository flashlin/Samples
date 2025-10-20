#!/bin/bash

# Build and pack T1.EfCodeFirstGenerateCli
# This script builds the project and creates a NuGet package

set -e

echo "==================================="
echo "Building T1.EfCodeFirstGenerateCli"
echo "==================================="

cd T1.EfCodeFirstGenerateCli

# Clean previous builds
echo "Cleaning previous builds..."
dotnet clean

# Restore dependencies
echo "Restoring dependencies..."
dotnet restore

# Build the project
echo "Building project..."
dotnet build -c Release

# Pack the NuGet package
echo "Creating NuGet package..."
dotnet pack -c Release

echo ""
echo "==================================="
echo "Build and pack completed successfully!"
echo "==================================="
echo ""
echo "Package location:"
ls -lh bin/Release/*.nupkg

echo ""
echo "To test the package locally:"
echo "  dotnet add package T1.EfCodeFirstGenerateCli --source ./bin/Release"
echo ""
echo "To publish to NuGet.org:"
echo "  dotnet nuget push bin/Release/T1.EfCodeFirstGenerateCli.1.0.1.nupkg --api-key YOUR_API_KEY --source https://api.nuget.org/v3/index.json"

