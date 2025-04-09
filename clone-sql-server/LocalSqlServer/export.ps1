param(
    [Parameter(Mandatory=$true)]
    [string]$SERVER,
    [string]$SQL_VERSION = "160" # SQL Server 2022 對應 version 是 160
)

# https://learn.microsoft.com/en-us/sql/tools/sqlpackage/sqlpackage-download?view=sql-server-ver16
# Install-Module -Name SqlServer -Force
Write-Host "$SERVER"

# 輸出資料夾
$OUTPUT_DIR = "./exports"
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

# 取得所有 user database 名稱（排除系統資料庫）
Write-Host "🔍 正在取得所有 user databases..."
$query = "SET NOCOUNT ON; SELECT name FROM sys.databases WHERE name NOT IN ('master','tempdb','model','msdb')"
$DATABASES = Invoke-Sqlcmd -ServerInstance $SERVER -Query $query -TrustServerCertificate | Select-Object -ExpandProperty name

foreach ($DB in $DATABASES) {
    Write-Host "📦 導出資料庫：$DB"

    # 取得使用到 Synonym 的預存程序清單
    $spWithSynonymQuery = @"
SELECT DISTINCT 
    QUOTENAME(OBJECT_SCHEMA_NAME(p.object_id)) + '.' + QUOTENAME(p.name) as ProcedureName
FROM sys.sql_modules m
INNER JOIN sys.procedures p ON m.object_id = p.object_id
INNER JOIN sys.synonyms s ON m.definition LIKE '%' + s.name + '%'
"@
    
    $spWithSynonyms = Invoke-Sqlcmd -ServerInstance $SERVER -Database $DB -Query $spWithSynonymQuery -TrustServerCertificate | Select-Object -ExpandProperty ProcedureName
    
    # 建立排除物件清單
    $excludeObjects = $spWithSynonyms -join ';'
    
    $OUTPUT_FILE = Join-Path $OUTPUT_DIR "Create_${DB}.bacpac"

    # 使用 SqlPackage.exe 導出資料庫
    & SqlPackage /Action:Export `
        /SourceServerName:$SERVER `
        /SourceDatabaseName:$DB `
        /TargetFile:$OUTPUT_FILE `
        /Properties:CompressionOption=Fast `
        /Properties:VerifyExtraction=True `
        /Properties:CommandTimeout=0 `
        /Properties:DatabaseLockTimeout=60 `
        /Properties:ExcludeObjectTypes=Synonyms `
        /Properties:ExcludeObjects=$excludeObjects `
        /SourceTrustServerCertificate:True

    Write-Host "✅ 已匯出：$OUTPUT_FILE"
    if ($spWithSynonyms) {
        Write-Host "⚠️ 已排除以下使用 Synonym 的預存程序："
        $spWithSynonyms | ForEach-Object { Write-Host "   - $_" }
    }
}

Write-Host "🎉 所有資料庫已成功匯出至 $OUTPUT_DIR" 