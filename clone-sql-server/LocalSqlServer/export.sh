#!/bin/bash

# 資料庫連線資訊
SERVER="localhost"
USERNAME="sa"
PASSWORD="YourStrongPassword123"
SQL_VERSION="160" # SQL Server 2022 對應 version 是 160

# 輸出資料夾
OUTPUT_DIR="./exports"
mkdir -p "$OUTPUT_DIR"

# 取得所有 user database 名稱（排除系統資料庫）
echo "🔍 正在取得所有 user databases..."
DATABASES=$(sqlcmd -S "$SERVER" -U "$USERNAME" -P "$PASSWORD" -Q "SET NOCOUNT ON; SELECT name FROM sys.databases WHERE name NOT IN ('master','tempdb','model','msdb')" -h -1)

# 去除空白列
DATABASES=$(echo "$DATABASES" | sed '/^\s*$/d')

for DB in $DATABASES; do
    echo "📦 導出資料庫：$DB"

    OUTPUT_FILE="${OUTPUT_DIR}/Create_${DB}.sql"

    SqlPackage /Action:Script \
        /SourceServerName:"$SERVER" \
        /SourceDatabaseName:"$DB" \
        /TargetFile:"$OUTPUT_FILE" \
        /SourceUser:"$USERNAME" \
        /SourcePassword:"$PASSWORD" \
        /p:ExtractAllTableData=False \
        /p:ScriptDatabaseOptions=True \
        /p:ScriptDrops=False \
        /p:IncludeCompositeObjects=True \
        /p:ScriptUseDatabase=True \
        /p:IncludeTransactionalScripts=False \
        /p:TargetServerVersion=SqlServer$SQL_VERSION

    echo "✅ 已匯出：$OUTPUT_FILE"
done

echo "🎉 所有資料庫已成功匯出至 $OUTPUT_DIR"
