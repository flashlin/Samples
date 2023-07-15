param(
    [string]$searchPattern
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

function DisplayDirs {
    param(
        [string[]]$dirs,
        [string]$pattern
    )
    $index = 0
    $dirs | ForEach-Object {
        $name = $_
        #Write-Host "$($index): $($name)"
        Write-Host "$($index): " -NoNewline
        WriteHostColor "$($name)" $pattern
        Write-Host ""
        $index += 1
    }
}

if( "" -eq $searchPattern ) {
    $jj = Get-Content -Path "D:\Demo\jj.txt"
    $searchPattern = $jj[0]
    $folderNames = $jj[1..($jj.Length - 1)]
    DisplayDirs $folderNames $searchPattern
    return
}

$folderPattern = $searchPattern
$regexPattern = "[\[\]\^\$\.\!\=]"
if (-Not ($searchPattern -match $regexPattern)) {
    $folderPattern = "^.*$searchPattern.*$"
}


$result = & es -name-color green /ad -regex $folderPattern
if ($result) {
    $folderNames = $result -split [Environment]::NewLine
    $folderNames = $folderNames | Where-Object { $_ -notmatch "^C:" } 
    $results = @( $searchPattern )
    $results += $folderNames
    $results | Set-Content -Path "D:\demo\jj.txt"
    DisplayDirs $folderNames $searchPattern
}
#Get-ChildItem -Directory | Where-Object { $_.Name -match $folderPattern } | ForEach-Object { $_.Name } | fzf

