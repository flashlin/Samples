#!/bin/bash
setenforce 0
cd /app/Database
./entrypoint.sh
cd /app
dotnet QueryWeb.dll