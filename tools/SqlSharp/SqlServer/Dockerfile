# Dockerfile
FROM mcr.microsoft.com/mssql/server:2022-latest

# 複製初始化腳本到映像中
COPY init.sql /usr/src/app/init.sql

# 設定 CMD 執行 SQLCMD 在容器啟動後執行 init.sql
CMD ["bash", "-c", "sleep 60; /opt/mssql-tools/bin/sqlcmd -S localhost -U SA -P 'YourPassword123' -i /usr/src/app/init.sql"]