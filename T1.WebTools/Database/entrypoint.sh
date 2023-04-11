#!/bin/bash
export ACCEPT_EULA=Y
export MSSQL_SA_PASSWORD=Passw0rd!
export MSSQL_PID=Developer
export MSSQL_TCP_PORT=1433
export MSSQL_AGENT_ENABLED=true 
/opt/mssql/bin/sqlservr & 
./init.sh