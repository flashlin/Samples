param(
    [string]$selectIndex
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

if( "" -eq $selectIndex ) {
    $selectIndex = 0
}

$selectIndex = [int]::Parse($selectIndex) + 1
$result = Get-Content -Path "D:\Demo\jj.txt"
$dir = $result[$selectIndex]

$searchPattern = $result[0]
WriteHostColor "$dir" $searchPattern
Write-Host ""

Set-Location -Path $dir

# copy to clipboard
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Clipboard]::SetText($dir)


#Get-ChildItem -Directory | Where-Object { $_.Name -match $folderPattern } | ForEach-Object { $_.Name } | fzf

