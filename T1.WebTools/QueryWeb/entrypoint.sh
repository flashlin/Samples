#!/bin/bash
cd /app/Database
./entrypoint.sh
cd /app
dotnet QueryWeb.dll