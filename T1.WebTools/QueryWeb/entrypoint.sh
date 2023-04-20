#!/bin/bash
echo "starting Database entrypoint"
su - mssql -c "cd /app/Database && /app/Database/entrypoint.sh"
cd /app
#apt-get update
#apt-get install net-tools
#netstat -tln
echo "===================="
echo " QueryWeb "
echo "===================="
dotnet QueryWeb.dll