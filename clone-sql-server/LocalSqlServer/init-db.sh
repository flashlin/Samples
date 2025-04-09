#!/bin/bash

# 啟動 SQL Server
/opt/mssql/bin/sqlservr &

# 等待 SQL Server 啟動
echo "======================"
echo "等待 SQL Server 啟動..."
echo "======================"
for i in {1..60}; do
    echo ".........."
    if /opt/mssql-tools18/bin/sqlcmd -S "localhost,1433;TrustServerCertificate=yes" -U SA -P $SA_PASSWORD -Q "SELECT 1" -C -N -t 30 &> /dev/null; then
        echo "SQL Server 已啟動"
        break
    fi
    sleep 1
done

# 執行資料庫初始化腳本
echo "====================="
echo "執行資料庫初始化腳本..."
echo "====================="
/opt/mssql-tools18/bin/sqlcmd -S "localhost,1433;TrustServerCertificate=yes" -U SA -P $SA_PASSWORD -i CreateDatabase.sql -C -N -t 30

# 保持容器運行
tail -f /dev/null 