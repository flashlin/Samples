param(
    [string]$selectIndex
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

if( "" -eq $selectIndex ) {
    $selectIndex = 0
}

$dirs = Get-Content -Path "d:\demo\jj.txt"
$dir = $dirs[$selectIndex]

Set-Location -Path $dir

#Get-ChildItem -Directory | Where-Object { $_.Name -match $folderPattern } | ForEach-Object { $_.Name } | fzf

