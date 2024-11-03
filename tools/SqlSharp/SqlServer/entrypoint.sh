#!/bin/bash
set -x

# 啟動 SQL Server
/opt/mssql/bin/sqlservr &

# 將標準輸出和錯誤輸出重定向到日誌文件
# exec > /usr/src/app/entrypoint.log 2>&1

# 等待 SQL Server 啟動完成
echo "等待 SQL Server 啟動..."
# 使用循環來檢查 SQL Server 是否已經可以接受連接
for i in {30..0}; do
    if sqlcmd -S localhost -U sa -P 'YourStrong!Passw0rd' -C -Q 'SELECT 1' > /dev/null 2>&1; then
        echo "SQL Server 啟動完成!"
        break
    fi
    echo "SQL Server 尚未啟動... 還剩下 $i 秒"
    sleep 1
done

echo "執行初始化腳本"
sqlcmd -S localhost -U sa -P 'YourStrong!Passw0rd' -C -i /usr/src/app/init.sql

#docker exec -it sql-server-db /bin/bash
#cat /usr/src/app/entrypoint.log

echo "保持容器運行"
wait