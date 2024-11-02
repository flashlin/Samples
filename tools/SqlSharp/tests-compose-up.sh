#!/bin/bash

# 啟動 Docker Compose 並以背景模式運行
docker-compose up --build -d

sudo netstat -tulnp

#echo "等待服務啟動..."
#sleep 5
docker-compose ps

docker logs sql-server-db
