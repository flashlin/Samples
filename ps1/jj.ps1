param(
    [string]$folderPattern
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

function DisplayDirs {
    param(
        [string[]]$dirs
    )
    $index = 0
    $dirs | ForEach-Object {
        Write-Host "$($index): $($_)"
        $index += 1
    }
}

if( "" -eq $folderPattern ) {
    $folderNames = Get-Content -Path "D:\Demo\jj.txt"
    DisplayDirs $folderNames
    return
}

$regexPattern = "[\[\]\^\$\.\!\=]"
if (-Not ($folderPattern -match $regexPattern)) {
    $folderPattern = "^.*$folderPattern.*$"
}

$result = & es -name-color green /ad -regex $folderPattern
if ($result) {
    $folderNames = $result -split [Environment]::NewLine
    $folderNames | Where-Object { $_ -notmatch "^C:" } 
    $folderNames | Set-Content -Path "D:\demo\jj.txt"
    DisplayDirs $folderNames
}
#Get-ChildItem -Directory | Where-Object { $_.Name -match $folderPattern } | ForEach-Object { $_.Name } | fzf

