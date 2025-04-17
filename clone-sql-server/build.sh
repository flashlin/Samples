#!/bin/bash

echo "Exporting database structure..."
dotnet build
./CloneSqlServer/bin/Debug/net9.0/CloneSqlServer 127.0.0.1,1433 "LocalSqlServer"

# Build Docker image
echo "Building Docker image..."
cd LocalSqlServer
./build-image.sh

cd ..