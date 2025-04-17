#!/bin/bash

# 定義容器名稱變數
container_name="local_sql_server"

cd LocalSqlServer
./build-image.sh
cd ..

# 移除舊容器
docker rm -f $container_name

# 運行新容器
docker run -d --name $container_name -p 4433:1433 -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=YourStrongPassw0rd!" local-sql-server 

# 顯示容器日誌
echo "docker logs $container_name"
docker logs $container_name
