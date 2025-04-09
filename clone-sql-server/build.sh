#!/bin/bash

# 檢查是否提供了 SQL Server 實例參數
if [ $# -eq 0 ]; then
    echo "請提供 SQL Server 實例參數"
    echo "使用方式: ./build.sh 127.0.0.1:3390"
    exit 1
fi

SQL_SERVER_INSTANCE="$1"

echo "開始執行資料庫克隆流程..."

# 執行 CloneSqlServer 程式
echo "正在導出資料庫結構..."
./CloneSqlServer/bin/Debug/net9.0/CloneSqlServer "$SQL_SERVER_INSTANCE" "LocalSqlServer"

# 檢查是否成功生成 SQL 檔案
if [ ! -f "CreateDatabase.sql" ]; then
    echo "錯誤：未能生成 CreateDatabase.sql 檔案"
    exit 1
fi

# 建立 Docker 映像
echo "正在建立 Docker 映像..."
docker build -t sbo-sql-server .

echo "完成！Docker 映像 'sbo-sql-server' 已建立" 