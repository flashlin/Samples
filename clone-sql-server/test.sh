#!/bin/bash

# Define variables
DOCKER_IMAGE="test-sql-server"
CONTAINER_NAME="test-sql-server"
export SQL_SA_PASSWORD="YourStrongPassw0rd!"

# Define action list
action_list=("rebuild" "run")

# Show fzf menu and get selected action
selected_action=$(printf '%s\n' "${action_list[@]}" | fzf --prompt="Please select action: ")

# Check if rebuild is selected
if [ "$selected_action" = "rebuild" ]; then
    echo "Rebuilding Docker image..."
    docker rmi -f $DOCKER_IMAGE
fi

# Check if Docker image exists
if ! docker images $DOCKER_IMAGE | grep -q $DOCKER_IMAGE; then
    echo "Docker image '$DOCKER_IMAGE' not found. Building..."
    cd TestSqlServer
    docker build -t $DOCKER_IMAGE --build-arg SQL_SA_PASSWORD=$SQL_SA_PASSWORD .
    cd ..
else
    echo "Docker image '$DOCKER_IMAGE' found."
fi

docker rm -f $CONTAINER_NAME

# Run the container
docker run -d --name $CONTAINER_NAME -p 1433:1433 \
    -e "ACCEPT_EULA=Y" \
    -e "SQL_SA_PASSWORD=$SQL_SA_PASSWORD" \
    $DOCKER_IMAGE

echo "SQL Server container is starting..."
echo "Waiting for container to be ready..."
sleep 10s
echo "Container should be ready now. You can connect to SQL Server at localhost,1433"
echo "SA Password: $SQL_SA_PASSWORD"

docker logs $CONTAINER_NAME
