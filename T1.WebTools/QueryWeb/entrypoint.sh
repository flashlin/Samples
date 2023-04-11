#!/bin/bash
echo "starting Database entrypoint"
su - mssql -c "cd /app/Database && /app/Database/entrypoint.sh"
cd /app
echo "===================="
echo " QueryWeb "
echo "===================="
dotnet QueryWeb.dll