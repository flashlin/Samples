#!/bin/bash

echo "Exporting database structure..."
dotnet build
./CloneSqlServer/bin/Debug/net9.0/CloneSqlServer devdb.coreop.net "LocalSqlServer"

# Build Docker image
echo "Building Docker image..."
cd LocalSqlServer
./build-image.sh
cd ..