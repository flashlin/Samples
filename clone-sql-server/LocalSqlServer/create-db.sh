#!/bin/bash

# Start SQL Server
/opt/mssql/bin/sqlservr &

echo "======================"
echo "Waiting for SQL Server to start..."
echo "======================"

# Add initial wait time to allow SQL Server to complete initialization
sleep 10

for i in {1..90}; do
    echo "Attempt $i to connect..."
    if /opt/mssql-tools18/bin/sqlcmd \
        -S 127.0.0.1 \
        -U SA \
        -P $SA_PASSWORD \
        -Q "SELECT @@VERSION" \
        -C -N -t 30 \
        2>&1; then
        echo "SQL Server started successfully"
        break
    fi
    echo "Waiting for SQL Server to be ready..."
    sleep 2
done

# Execute database initialization script
echo "====================="
echo "Executing database initialization script..."
echo "====================="
/opt/mssql-tools18/bin/sqlcmd \
    -S 127.0.0.1 \
    -U SA \
    -P $SA_PASSWORD \
    -i CreateDatabase.sql \
    -C -N -t 30
