param(
    [Parameter(Mandatory=$true)]
    [string]$SqlServerInstance
)

Write-Host "開始執行資料庫克隆流程..."

# 執行 CloneSqlServer 程式
Write-Host "正在導出資料庫結構..."
.\CloneSqlServer\bin\Debug\net9.0\CloneSqlServer.exe $SqlServerInstance

# 檢查是否成功生成 SQL 檔案
if (!(Test-Path "CreateDatabase.sql")) {
    Write-Host "錯誤：未能生成 CreateDatabase.sql 檔案"
    exit 1
}

# 建立 Docker 映像
Write-Host "正在建立 Docker 映像..."
docker build -t sbo-sql-server .

Write-Host "完成！Docker 映像 'sbo-sql-server' 已建立" 