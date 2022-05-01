#docker build -t sql-localdb-test -f .\SqlLocalDataTests\Dockerfile . # --no-cache
Write-Host "running tests" -ForegroundColor Green
#docker run --name test sql-localdb-test
#docker run --rm -v ${pwd}:/SqlLocalDataTests -w /SqlLocalDataTests mcr.microsoft.com/dotnet/sdk:6.0 dotnet test --logger:trx


docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=<YourStrong@Passw0rd>" `
   -p 1433:2433 --name sql1 --hostname sql1 `
   -d mcr.microsoft.com/mssql/server:2019-latest

docker exec -it sql1 "bash"
