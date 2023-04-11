#!/bin/bash
USER root
cd /app/Database
echo "starting Database entrypoint"
./entrypoint.sh
cd /app
dotnet QueryWeb.dll