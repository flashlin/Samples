#!/bin/bash

# 啟動 SQL Server
/opt/mssql/bin/sqlservr &

# 等待 SQL Server 啟動
echo "等待 SQL Server 啟動..."
for i in {1..60}; do
    if /opt/mssql-tools18/bin/sqlcmd -S localhost -U SA -P $SA_PASSWORD -Q "SELECT 1" &> /dev/null; then
        echo "SQL Server 已啟動"
        break
    fi
    sleep 1
done

# 執行資料庫初始化腳本
echo "執行資料庫初始化腳本..."
/opt/mssql-tools18/bin/sqlcmd -S localhost -U SA -P $SA_PASSWORD -i CreateDatabase.sql

# 保持容器運行
tail -f /dev/null 