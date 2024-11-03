#!/bin/bash
sudo chown -R 10001:10001 /home/flash/mssql
sudo chmod -R 777 /home/flash/mssql

docker rm -f sql-server-db
docker-compose down

# 啟動 Docker Compose 並以背景模式運行
docker-compose up --build -d

#echo "等待服務啟動..."
#sleep 5
docker-compose ps

# 取得測試服務的容器 ID
container_id=$(docker-compose ps -q sql-sharp-tests)
echo "測試服務容器 ID: $container_id"

# 檢查容器執行狀態，並等待測試完成
echo "等待測試完成..."
docker wait "$container_id"

# 將測試報告從容器中複製出來
docker cp "$container_id":/app/TestResults.trx ./TestResults.trx

# 確認測試結果是否存在
if [[ -f "./TestResults.trx" ]]; then
    echo "測試已完成，結果儲存在 TestResults.trx"
    cat ./TestResults.trx
else
    echo "測試未能成功執行或無法生成測試報告。"
    docker logs sql-server-db
    echo "------------------------------------------------------------------"
    docker logs sql-sharp-tests
fi

# 停止並刪除容器
docker-compose down