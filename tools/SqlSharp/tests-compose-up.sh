#!/bin/bash

# 啟動 Docker Compose 並以背景模式運行
docker-compose up --build -d

#echo "等待服務啟動..."
docker exec -it sql-server-db sh -c 'sqlcmd -S localhost -U sa -P '\''YourStrong!Passw0rd'\'' -C -Q '\''select * from Customer where name > '\'''\''   '\'''