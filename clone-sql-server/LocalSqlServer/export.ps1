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
$TEMP_DIR = "./temp"
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null

# 取得所有 user database 名稱（排除系統資料庫）
Write-Host "🔍 正在取得所有 user databases..."
$query = "SET NOCOUNT ON; SELECT name FROM sys.databases WHERE name NOT IN ('master','tempdb','model','msdb')"
$DATABASES = Invoke-Sqlcmd -ServerInstance $SERVER -Query $query -TrustServerCertificate | Select-Object -ExpandProperty name

foreach ($DB in $DATABASES) {
    Write-Host "📦 處理資料庫：$DB"

    # 檢查是否有不支援的元素
    $checkUnsupportedQuery = @"
-- 檢查 Windows 使用者和群組
SELECT COUNT(*) as WindowsUsers
FROM sys.database_principals 
WHERE type_desc IN ('WINDOWS_USER', 'WINDOWS_GROUP');

-- 檢查 Synonyms
SELECT COUNT(*) as SynonymCount
FROM sys.synonyms;
"@

    $unsupportedCounts = Invoke-Sqlcmd -ServerInstance $SERVER -Database $DB -Query $checkUnsupportedQuery -TrustServerCertificate
    $hasWindowsUsers = $unsupportedCounts[0].WindowsUsers -gt 0
    $hasSynonyms = $unsupportedCounts[1].SynonymCount -gt 0

    $needsCleaning = $hasWindowsUsers -or $hasSynonyms

    if ($needsCleaning) {
        Write-Host "⚠️ 發現不支援的元素："
        if ($hasWindowsUsers) { Write-Host "   - Windows 使用者/群組: $($unsupportedCounts[0].WindowsUsers) 個" }
        if ($hasSynonyms) { Write-Host "   - Synonyms: $($unsupportedCounts[1].SynonymCount) 個" }
        
        # 建立資料庫複本
        $tempDB = "${DB}_Export_Temp"
        Write-Host "正在建立暫存資料庫 $tempDB..."
        
        # 檢查是否已存在暫存資料庫，如果有就先刪除
        $dropTempDBQuery = "IF EXISTS(SELECT 1 FROM sys.databases WHERE name = '$tempDB') BEGIN ALTER DATABASE [$tempDB] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [$tempDB]; END"
        Invoke-Sqlcmd -ServerInstance $SERVER -Query $dropTempDBQuery -TrustServerCertificate

        # 取得資料庫檔案位置
        $dbFilesQuery = @"
SELECT 
    mf.physical_name,
    mf.name,
    mf.type_desc
FROM sys.master_files mf
INNER JOIN sys.databases d ON d.database_id = mf.database_id
WHERE d.name = '$DB'
"@
        $dbFiles = Invoke-Sqlcmd -ServerInstance $SERVER -Query $dbFilesQuery -TrustServerCertificate

        # 準備備份和還原的路徑
        $backupFile = Join-Path $TEMP_DIR "${DB}_temp.bak"
        
        # 備份原始資料庫
        Write-Host "正在備份原始資料庫..."
        $backupQuery = "BACKUP DATABASE [$DB] TO DISK = N'$backupFile' WITH INIT, COMPRESSION"
        Invoke-Sqlcmd -ServerInstance $SERVER -Query $backupQuery -TrustServerCertificate

        # 準備還原命令
        $restoreQuery = @"
RESTORE DATABASE [$tempDB] 
FROM DISK = N'$backupFile'
WITH 
"@
        # 為每個檔案準備新的位置
        $fileList = @()
        foreach ($file in $dbFiles) {
            $originalPath = Split-Path -Path $file.physical_name
            $fileName = Split-Path -Path $file.physical_name -Leaf
            $newFileName = $fileName.Replace($DB, $tempDB)
            $newPath = Join-Path $originalPath $newFileName
            $fileList += "MOVE N'$($file.name)' TO N'$newPath'"
        }
        $restoreQuery += ($fileList -join ",`n") + ",`nREPLACE"

        # 還原為新的資料庫
        Write-Host "正在還原為暫存資料庫..."
        Invoke-Sqlcmd -ServerInstance $SERVER -Query $restoreQuery -TrustServerCertificate

        # 刪除備份檔案
        Remove-Item -Path $backupFile -Force

        # 在複本上執行清理
        Write-Host "正在清理暫存資料庫..."
        $cleanupQuery = @"
-- 移除 Windows 使用者的所有權限
DECLARE @sql NVARCHAR(MAX) = ''
SELECT @sql += 'REVOKE ALL FROM ' + QUOTENAME(name) + '; '
FROM sys.database_principals 
WHERE type_desc IN ('WINDOWS_USER', 'WINDOWS_GROUP')
IF LEN(@sql) > 0
    EXEC sp_executesql @sql

-- 移除 Windows 使用者的所有角色成員資格
SELECT @sql = ''
SELECT @sql += 'ALTER ROLE ' + QUOTENAME(r.name) + ' DROP MEMBER ' + QUOTENAME(m.name) + '; '
FROM sys.database_role_members rm
JOIN sys.database_principals r ON r.principal_id = rm.role_principal_id
JOIN sys.database_principals m ON m.principal_id = rm.member_principal_id
WHERE m.type_desc IN ('WINDOWS_USER', 'WINDOWS_GROUP')
IF LEN(@sql) > 0
    EXEC sp_executesql @sql

-- 移除所有 Windows 使用者
SELECT @sql = ''
SELECT @sql += 'DROP USER ' + QUOTENAME(name) + '; '
FROM sys.database_principals 
WHERE type_desc IN ('WINDOWS_USER', 'WINDOWS_GROUP')
IF LEN(@sql) > 0
    EXEC sp_executesql @sql

-- 移除所有 Synonyms
SELECT @sql = ''
SELECT @sql += 'DROP SYNONYM ' + QUOTENAME(SCHEMA_NAME(schema_id)) + '.' + QUOTENAME(name) + '; '
FROM sys.synonyms
IF LEN(@sql) > 0
    EXEC sp_executesql @sql
"@
        Invoke-Sqlcmd -ServerInstance $SERVER -Database $tempDB -Query $cleanupQuery -TrustServerCertificate
        
        # 從清理後的複本導出
        $targetDB = $tempDB
    } else {
        Write-Host "✅ 未發現不支援的元素，將直接從原始資料庫導出"
        $targetDB = $DB
    }

    $OUTPUT_FILE = Join-Path $OUTPUT_DIR "Create_${DB}.bacpac"

    # 使用 SqlPackage.exe 導出資料庫
    Write-Host "正在導出資料庫..."
    & SqlPackage /Action:Export `
        /SourceServerName:$SERVER `
        /SourceDatabaseName:$targetDB `
        /TargetFile:$OUTPUT_FILE `
        /Properties:CompressionOption=Fast `
        /Properties:VerifyExtraction=True `
        /Properties:CommandTimeout=0 `
        /Properties:DatabaseLockTimeout=60 `
        /Properties:Storage=Memory `
        /SourceTrustServerCertificate:True

    Write-Host "✅ 已匯出：$OUTPUT_FILE"

    # 如果使用了暫存資料庫，清理它
    if ($needsCleaning) {
        Write-Host "正在清理暫存資料庫..."
        $dropTempDBQuery = "ALTER DATABASE [$tempDB] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [$tempDB]"
        Invoke-Sqlcmd -ServerInstance $SERVER -Query $dropTempDBQuery -TrustServerCertificate
    }
}

# 清理臨時目錄
Remove-Item -Path $TEMP_DIR -Force -Recurse
Write-Host "🎉 所有資料庫已成功匯出至 $OUTPUT_DIR" 