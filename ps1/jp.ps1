param (
    [Parameter(Position = 0)]
    [string]$pattern = "",

    [Parameter(Mandatory=$false)]
    [switch]$add = $false

    # [Parameter(Position = 1)]
    # [string]$excludePattern = "\\node_modules,\\obj,\\packages,\\.git,\\.vs,\\.vscode"
)
Import-Module "$($env:psm1HOME)/Common.psm1" -Force
Import-Module "$($env:psm1HOME)/ConsoleCore.psm1" -Force

$workPath = (Get-Location).Path

$searcher = New-Object ConsoleCore.FolderSearcher

# if( $add ) {
if( "" -eq $pattern ) {
    $searcher.AddFolder($pwd)
    Write-Host "save '$pwd' path"
    return
}

$folders = $searcher.Find($pattern)
$idx = 0
$folders | ForEach-Object {
    $item = $_
    Write-Host -NoNewline "$idx "
    $item.WriteDisplayResult()
    Write-Host ""
    $idx += 1
}

if( $idx -eq 1 ) {
    $folder = $searcher.GetSearchResult(0)
    Write-Host $folder -ForegroundColor Green
    Set-Location $folder
    cpwd
    return
}

if( $idx -eq 0 ) {
    Write-Host "Not found"
    return
}
# Write-Host "Please type 'gg [number]' to choice above paths"
