#!/bin/bash

# 啟動 SQL Server
/opt/mssql/bin/sqlservr &

# 等待 SQL Server 啟動
echo "======================"
echo "等待 SQL Server 啟動..."
echo "======================"

# 增加初始等待時間，讓 SQL Server 有足夠時間完成初始化
sleep 10

for i in {1..90}; do
    echo "第 $i 次嘗試連接..."
    if /opt/mssql-tools18/bin/sqlcmd \
        -S 127.0.0.1 \
        -U SA \
        -P $SA_PASSWORD \
        -Q "SELECT @@VERSION" \
        -C -N -t 30 \
        2>&1; then
        echo "SQL Server 已成功啟動"
        break
    fi
    echo "等待 SQL Server 就緒..."
    sleep 2
done

# 執行資料庫初始化腳本
echo "====================="
echo "執行資料庫初始化腳本..."
echo "====================="
/opt/mssql-tools18/bin/sqlcmd \
    -S 127.0.0.1 \
    -U SA \
    -P $SA_PASSWORD \
    -i CreateDatabase.sql \
    -C -N -t 30

# 保持容器運行
tail -f /dev/null 