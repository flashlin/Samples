param(
    [parameter(Position = 0, ValueFromRemainingArguments = $true)]    
    [string[]]$searchPatterns
)
#Install-Package System.Data.SQLite
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

if( $searchPatterns.Length -gt 0 ) {
    $action = $searchPatterns[0]
}
Info "$action"

# Add-Type -AssemblyName System.Data.SQLite
Add-Type -Path "$($env:psm1HOME)/../sqlite/System.Data.SQLite.dll"
$path = "$($env:psm1HOME)/../sqlite"
if ($env:PATH -notlike "*$path*") {
    $env:PATH += ";$path"
}


$databasePath = "D:/Demo/folders.db"
$connectionString = "Data Source=$databasePath;Version=3;"

function ExecuteNonQuery {
    param(
        [string]$query
    )
    $command = $connection.CreateCommand()
    $command.CommandText = $query
    $affectedRows = $command.ExecuteNonQuery()
}

function FetchQuery {
    param(
        [string]$query,
        [scriptblock]$processRow
    )
    $command = $connection.CreateCommand()
    $command.CommandText = $query
    $reader = $command.ExecuteReader()
    $schemaTable = $reader.GetSchemaTable()
    $columnNames = @()
    foreach ($row in $schemaTable.Rows) {
        $columnName = $row["ColumnName"]
        $columnNames += $columnName
    }
    while ($reader.Read()) {
        $row = @{}
        foreach ($columnName in $columnNames) {
            $columnValue = $reader[$columnName]
            AddObjectProperty $row $columnName $columnValue
            # $row | Add-Member -MemberType NoteProperty -Name $columnName -Value $columnValue
        }
        # Write-Host (DumpObject $row)
        $processRow.Invoke($row)
    }
}

function ExecuteQuery {
    param(
        [string]$query
    )
    $command = $connection.CreateCommand()
    $command.CommandText = $query
    $reader = $command.ExecuteReader()
    $schemaTable = $reader.GetSchemaTable()
    $columnNames = @()
    foreach ($row in $schemaTable.Rows) {
        $columnName = $row["ColumnName"]
        $columnNames += $columnName
    }
    $result = @()
    while ($reader.Read()) {
        $row = @{}
        foreach ($columnName in $columnNames) {
            $columnValue = $reader[$columnName]
            $row | Add-Member -MemberType NoteProperty -Name $columnName -Value $columnValue
        }
        $result += $row
    }
    return $result
}

function UpsertPath {
    param (
        [string]$path
    )
    $sql = "INSERT OR IGNORE INTO directory (dirName) VALUES ('$path')"
    ExecuteNonQuery $sql
}

$excludeDirs = @(
    '\.git', '\.vscode', '\.idea', '\.github', 
    'bin', 'obj', 'runs',
    'dist', 'build', 'out', 'output', 'logs', 'temp', 'tmp',
    'node_modules', 'packages', 
    '__pycache__',
    '.chainlit'
)

function IsExcludePath {
    param(
        [string]$path
    )
    foreach($pattern in $excludeDirs) {
        if( $path -match "(?<=$pattern)" ) {
            #Write-Host "$pattern  '$path'"
            return $true
        }
    }
    return $false
}

function QueryDirectories {
    param(
        [string]$directoryPath = "./"
    )
    process {
        $childDirectories = Get-ChildItem -Path $directoryPath -Directory
        foreach ($directory in $childDirectories) {
            if( IsExcludePath $directory.FullName ) {
                continue
            }
            $directory.FullName
            $subDirs = QueryDirectories $directory.FullName
            foreach($subDir in $subDirs) {
                $subDir
            }
        }
    }
}


function FilterDirectoriesByPattern {
    param(
        [string[]]$paths,
        [string]$pattern
    )
    return $paths | Where-Object { $_ -match $pattern } 
}

function FilterDirectories {
    param(
        [string[]]$paths,
        [string[]]$searchPatterns
    )
    for ($i = 1; $i -lt $searchPatterns.Length; $i++) {
        $pattern = $searchPatterns[$i]
        $paths = FilterDirectoriesByPattern $paths $pattern
    }
    return $paths
}

function JumpDirectory {
    param(
        [string]$path
    )
    WriteHostColor "$path" $searchPatterns
    Write-Host ""
    Set-Location -Path $path
    CopyToClipboard $path
}


# 建立連線
Info "open folders.db"
$connection = New-Object -TypeName System.Data.SQLite.SQLiteConnection
$connection.ConnectionString = $connectionString
$connection.Open()

$sql = "CREATE TABLE IF NOT EXISTS directory (dirName text)"
ExecuteNonQuery $sql
$sql = "CREATE INDEX IF NOT EXISTS idx_directory ON directory (dirName)"
ExecuteNonQuery $sql
$sql = "CREATE TABLE IF NOT EXISTS history (dirName text, createdOn default current_timestamp)"
ExecuteNonQuery $sql
$sql = "CREATE INDEX IF NOT EXISTS idx_history ON history (createdOn)"
ExecuteNonQuery $sql

if( "--a" -eq $action ){
    Write-Host "Upsert directories"
    foreach($dir in QueryDirectories) {
        UpsertPath $dir
        Write-Host $dir
    }
    return
}

if( "--clean" -eq $action ){
    Write-Host "Clean directories"
    $sql = "SELECT dirName FROM directory"

    $process = {
        param(
            [System.Collections.Hashtable]$row
        )
        $dirName = $row.dirName
        if( -not (Test-Path -Path $dirName -PathType Container) ) {
            Info "$dirName not exists"
            $sql = "DELETE FROM directory WHERE dirName = '$dirname'"
            ExecuteNonQuery $sql
            return
        }
        foreach($pattern in $excludeDirs) {
            if( $dirName -match $pattern ){
                WriteHostColorText $dirName @($pattern)
                $sql = "DELETE FROM directory WHERE dirName = '$dirname'"
                ExecuteNonQuery $sql
            }
        }
    }

    FetchQuery $sql $process
    return
}

if( "" -ne $action ) {
    $dirName = $action
    $sql = "SELECT dirName FROM directory WHERE dirName like '%$dirName%'"
    $paths = ExecuteQuery $sql | ForEach-Object { $_.dirName }
    $paths = FilterDirectories $paths $searchPatterns
    $paths = $paths | Sort-Object -Property Length, Name -Descending

    $result = @( $searchPatterns )
    $result += $paths
    $result | Set-Content -Path "D:\demo\jj.txt"

    if( $paths.Length -eq 0 ) {
        Write-Host "Not found"
        return
    }

    if( $paths.Length -eq 1 ) {
        JumpDirectory $paths[0]
        return
    }

    #foreach($path in $paths) {
    #    WriteHostColorText $path $searchPatterns
    #    #Write-Host $path
    #}

    $selectedPath = PromptList $paths
    if( "" -eq $selectedPath ) {
        return
    }

    JumpDirectory $selectedPath
    return
}

$connection.Close()
