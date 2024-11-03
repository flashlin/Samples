docker rm -f sql-server-db
docker rmi -f sql-server-db
docker build -t sql-server-db -f ./SqlServer/Dockerfile ./SqlServer

docker run -it --name sql-server-db -p 14333:1433 -d sql-server-db

sleep 30
docker logs sql-server-db
#docker exec -it sql-server-db /bin/bash


docker exec sql-server-db sh -c 'sqlcmd -S localhost -U sa -C -P '\''YourStrong!Passw0rd'\'' -Q '\''SELECT name FROM sys.databases'\'''