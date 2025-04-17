#!/bin/bash

echo "Building Docker image..."
docker build -f Dockerfile -t local-sql-server .

echo "Done! Docker image 'local-sql-server' has been created" 