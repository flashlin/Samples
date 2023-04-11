#!/bin/bash
cd /app/Database
echo "starting Database entrypoint"
/app/Database/entrypoint.sh
cd /app
dotnet QueryWeb.dll