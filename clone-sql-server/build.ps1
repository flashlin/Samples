Write-Host "Starting database cloning process..."
dotnet build

# Execute CloneSqlServer program
Write-Host "Exporting database structure..."
.\CloneSqlServer\bin\Debug\net9.0\CloneSqlServer.exe devdb.coreop.net "LocalSqlServer"

# Check if SQL file was successfully generated
if (!(Test-Path "LocalSqlServer/CreateDatabase.sql")) {
    Write-Host "Error: Failed to generate CreateDatabase.sql file"
    exit 1
}

# Check if SQL_SA_PASSWORD environment variable exists
if ([string]::IsNullOrEmpty($env:SQL_SA_PASSWORD)) {
    Write-Host "Error: SQL_SA_PASSWORD environment variable is not set"
    exit 1
}

# Build Docker image
Write-Host "Building Docker image..."
Set-Location LocalSqlServer
docker build --build-arg SA_PASSWORD=$env:SQL_SA_PASSWORD -t local-sql-server .

Set-Location ..
Write-Host "Done! Docker image 'local-sql-server' has been created"