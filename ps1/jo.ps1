param(
    [parameter(Position = 0, ValueFromRemainingArguments = $true)]    
    [string[]]$searchPatterns
)
#Install-Package System.Data.SQLite
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

if ( $searchPatterns.Length -gt 0 ) {
    $action = $searchPatterns[0]
}
Info "$action"

$sqlite = new-sqlite "D:/Demo/folders.db"

function UpsertPath {
    param (
        [string]$path
    )
    $sql = "INSERT OR IGNORE INTO directory (dirName) VALUES ('$path')"
    $sqlite.ExecuteNonQuery($sql)
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
    foreach ($pattern in $excludeDirs) {
        if ( $path -match "(?<=$pattern)" ) {
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
            if ( IsExcludePath $directory.FullName ) {
                continue
            }
            $directory.FullName
            $subDirs = QueryDirectories $directory.FullName
            foreach ($subDir in $subDirs) {
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
$sql = "CREATE TABLE IF NOT EXISTS directory (dirName text)"
$sqlite.ExecuteNonQuery($sql)
$sql = "CREATE INDEX IF NOT EXISTS idx_directory ON directory (dirName)"
$sqlite.ExecuteNonQuery($sql)
$sql = "CREATE TABLE IF NOT EXISTS history (dirName text, createdOn default current_timestamp)"
$sqlite.ExecuteNonQuery($sql)
$sql = "CREATE INDEX IF NOT EXISTS idx_history ON history (createdOn)"
$sqlite.ExecuteNonQuery($sql)

if ( "--a" -eq $action ) {
    Write-Host "Upsert directories"
    foreach ($dir in QueryDirectories) {
        UpsertPath $dir
        Write-Host $dir
    }
    return
}

if ( "--clean" -eq $action ) {
    Write-Host "Clean directories"
    $sql = "SELECT dirName FROM directory"

    $process = {
        param(
            [System.Collections.Hashtable]$row
        )
        $dirName = $row.dirName
        if ( -not (Test-Path -Path $dirName -PathType Container) ) {
            Info "$dirName not exists"
            $sql = "DELETE FROM directory WHERE dirName = '$dirname'"
            $sqlite.ExecuteNonQuery($sql)
            return
        }
        foreach ($pattern in $excludeDirs) {
            if ( $dirName -match $pattern ) {
                WriteHostColorText $dirName @($pattern)
                $sql = "DELETE FROM directory WHERE dirName = '$dirname'"
                $sqlite.ExecuteNonQuery($sql)
            }
        }
    }

    FetchQuery $sql $process
    return
}

if ( "" -ne $action ) {
    $dirName = $action
    $sql = "SELECT dirName FROM directory WHERE dirName like '%$dirName%'"
    $paths = $sqlite.ExecuteQuery($sql) | ForEach-Object { $_.dirName }
    $paths = FilterDirectories $paths $searchPatterns
    $paths = $paths | Sort-Object -Property Length, Name -Descending

    $result = @( $searchPatterns )
    $result += $paths
    $result | Set-Content -Path "D:\demo\jj.txt"

    if ( $paths.Length -eq 0 ) {
        Write-Host "Not found"
        return
    }

    if ( $paths.Length -eq 1 ) {
        JumpDirectory $paths[0]
        return
    }

    #foreach($path in $paths) {
    #    WriteHostColorText $path $searchPatterns
    #    #Write-Host $path
    #}

    $selectedPath = PromptList $paths
    if ( "" -eq $selectedPath ) {
        return
    }

    JumpDirectory $selectedPath
    return
}

$sqlite.Close()
