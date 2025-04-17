# 定義容器名稱變數
$containerName = "local_sql_server"

Set-Location -Path LocalSqlServer
.\build-image.sh
Set-Location -Path ..

# 移除舊容器
docker rm -f $containerName

# 運行新容器
docker run -d --name $containerName -p 4433:1433 -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=YourStrongPassw0rd!" local-sql-server

# 顯示容器日誌
Write-Host "docker logs $containerName"
docker logs $containerName 