param(
    [string]$folderPattern
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

if( "" -eq $folderPattern ) {
    Write-Host "jump directory script"
    Write-Host "jj ???"
    return
}

$regexPattern = "[\[\]\^\$\.\!\=]"
if (-Not ($folderPattern -match $regexPattern)) {
    $folderPattern = "^.*$folderPattern.*$"
}

$result = & es -name-color green /ad -regex $folderPattern
if ($result) {
    $folderNames = $result -split [Environment]::NewLine
    $itemsContent | Set-Content -Path "D:\demo\jj.txt"
    $index = 0
    $folderNames | Where-Object { $_ -notmatch "^C:" } 
    | ForEach-Object {
        Write-Host "$($index): $($_)"
        $index += 1
    }
}
#Get-ChildItem -Directory | Where-Object { $_.Name -match $folderPattern } | ForEach-Object { $_.Name } | fzf

