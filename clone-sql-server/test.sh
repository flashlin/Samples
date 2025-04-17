#!/bin/bash

# Define variables
DOCKER_IMAGE="test-sql-server"
CONTAINER_NAME="test-sql-server"
export SQL_SA_PASSWORD="YourStrongPassw0rd!"

# Check if Docker image exists
if ! docker images $DOCKER_IMAGE | grep -q $DOCKER_IMAGE; then
    echo "Docker image '$DOCKER_IMAGE' not found. Building..."
    cd TestSqlServer
    docker build -t $DOCKER_IMAGE .
    cd ..
else
    echo "Docker image '$DOCKER_IMAGE' found."
fi

# Run the container
docker run -d --name $CONTAINER_NAME -p 1433:1433 \
    -e "ACCEPT_EULA=Y" \
    -e "SA_PASSWORD=$SQL_SA_PASSWORD" \
    $DOCKER_IMAGE

echo "SQL Server container is starting..."
echo "Waiting for container to be ready..."
sleep 10s
echo "Container should be ready now. You can connect to SQL Server at localhost,1433"
echo "SA Password: $SQL_SA_PASSWORD"

docker logs $CONTAINER_NAME

./build.sh
