#!/bin/bash

# 檢查是否成功生成 SQL 檔案
if [ ! -f "CreateDatabase.sql" ]; then
    echo "錯誤：未能生成 CreateDatabase.sql 檔案"
    exit 1
fi

# 建立 Docker 映像
echo "正在建立 Docker 映像..."
docker build -t sbo-sql-server .

echo "完成！Docker 映像 'sbo-sql-server' 已建立" 