Install-Module -Name SqlServer -Force

# 資料庫連線資訊
$SERVER = "localhost"
$USERNAME = "sa"
$PASSWORD = "YourStrongPassword123"
$SQL_VERSION = "160" # SQL Server 2022 對應 version 是 160

# 輸出資料夾
$OUTPUT_DIR = "./exports"
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

# 取得所有 user database 名稱（排除系統資料庫）
Write-Host "🔍 正在取得所有 user databases..."
$query = "SET NOCOUNT ON; SELECT name FROM sys.databases WHERE name NOT IN ('master','tempdb','model','msdb')"
$DATABASES = Invoke-Sqlcmd -ServerInstance $SERVER -Username $USERNAME -Password $PASSWORD -Query $query | Select-Object -ExpandProperty name

foreach ($DB in $DATABASES) {
    Write-Host "📦 導出資料庫：$DB"

    $OUTPUT_FILE = Join-Path $OUTPUT_DIR "Create_${DB}.sql"

    # 使用 SqlPackage.exe 導出資料庫結構
    & SqlPackage /Action:Script `
        /SourceServerName:$SERVER `
        /SourceDatabaseName:$DB `
        /TargetFile:$OUTPUT_FILE `
        /SourceUser:$USERNAME `
        /SourcePassword:$PASSWORD `
        /p:ExtractAllTableData=False `
        /p:ScriptDatabaseOptions=True `
        /p:ScriptDrops=False `
        /p:IncludeCompositeObjects=True `
        /p:ScriptUseDatabase=True `
        /p:IncludeTransactionalScripts=False `
        /p:TargetServerVersion="SqlServer$SQL_VERSION"

    Write-Host "✅ 已匯出：$OUTPUT_FILE"
}

Write-Host "🎉 所有資料庫已成功匯出至 $OUTPUT_DIR" 