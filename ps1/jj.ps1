param(
    [string]$searchPattern
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

function JumpToFirstDir {
    param(
        [string[]]$dirs,
        [string]$pattern
    )
    if( $dirs.Length -eq 1 ) {
        $dir = $dirs[0]
        WriteHostColor "$dir" $pattern
        Write-Host ""
        Set-Location -Path $dir
    }
}

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


$notNamePatterns = @(
    "^C:",
    "\\node_modules",
    "\\.git\\",
    "\\.vscode\\",
    "\\.idea\\",
    "\\.vscode-insiders\\",
    "\\.history\\",
    "\\.cache",
    "\\packages",
    "\\bin",
    "\\pkgs\\",
    "\\site-packages",
    "\\dist"
)

function IsValidFolder {
    param(
        [string]$name
    )
    foreach ($namePattern in $notNamePatterns) {
        if ($name -match $namePattern) {
            return $False
        }
    }
    return $True
}

$result = & es -name-color green /ad -regex $folderPattern
if ($result) {
    $folderNames = $result -split [Environment]::NewLine
    
    $folderNames = $folderNames | Where-Object { IsValidFolder $_ } 
    #$folderNames = $folderNames | Sort-Object -Property { $_.Length } -Descending
    $folderNames = $folderNames | Sort-Object -Property Length, Name -Descending

    $results = @( $searchPattern )
    $results += $folderNames
    $results | Set-Content -Path "D:\demo\jj.txt"

    if( $results.Length -eq 2 ) {
        JumpToFirstDir $folderNames $searchPattern
        return
    }

    DisplayDirs $folderNames $searchPattern
}
#Get-ChildItem -Directory | Where-Object { $_.Name -match $folderPattern } | ForEach-Object { $_.Name } | fzf

