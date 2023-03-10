docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=Passw0rd!" `
   -p 4331:1433 --name sql1 --hostname sql1 `
   -d `
   mcr.microsoft.com/mssql/server:2019-latest