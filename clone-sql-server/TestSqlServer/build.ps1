# Build the SQL Server image
$imageName = "test-sql-server"
$containerName = "test_sql_server"

# Remove existing container if it exists
docker rm -f $containerName

# Build the image
docker build -t $imageName .

# Run the container
docker run -d `
    --name $containerName `
    -p 1433:1433 `
    -e "SA_PASSWORD=YourStrongPassw0rd!" `
    $imageName

# Show container logs
Write-Host "Container logs:"
docker logs $containerName 