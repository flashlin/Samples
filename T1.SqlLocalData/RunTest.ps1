docker build -t sql-localdb-test -f .\SqlLocalDataTests\Dockerfile-Test . # --no-cache
Write-Host "running tests" -ForegroundColor Green
#docker run --name test sql-localdb-test
#docker run --rm -v ${pwd}:/SqlLocalDataTests -w /SqlLocalDataTests mcr.microsoft.com/dotnet/sdk:6.0 dotnet test --logger:trx

docker-compose up
#docker ps -a
#docker logs <container name>
#docker-compose rm
