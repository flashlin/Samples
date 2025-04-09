#!/bin/bash

# 檢查是否在 WSL 環境中
if ! grep -q Microsoft /proc/version; then
    echo "錯誤：此腳本必須在 WSL 環境中運行"
    exit 1
fi

# 檢查是否提供了 SQL Server 實例參數
if [ $# -eq 0 ]; then
    echo "請提供 SQL Server 實例參數"
    echo "使用方式: ./build-for-wsl.sh <SQL_SERVER_INSTANCE>"
    exit 1
fi

SQL_SERVER_INSTANCE="$1"

echo "開始執行資料庫克隆流程..."

# 在 WSL 中使用 PowerShell 執行 CloneSqlServer
echo "正在導出資料庫結構..."
powershell.exe -Command ".\CloneSqlServer\bin\Debug\net9.0\CloneSqlServer.exe '$SQL_SERVER_INSTANCE' 'LocalSqlServer'"

# 檢查是否成功生成 SQL 檔案
if [ ! -f "CreateDatabase.sql" ]; then
    echo "錯誤：未能生成 CreateDatabase.sql 檔案"
    exit 1
fi

# 建立 Docker 映像
echo "正在建立 Docker 映像..."
cd LocalSqlServer
./build-image.sh
cd ..

echo "完成！Docker 映像已建立" 