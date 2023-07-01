param(
    [string]$folder
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force

$output = & es.exe "folder:$folder"
$folders = $output -split "`r`n"

if( $folders.Count -eq 1 ) {
    Info "Jump to $($folders[0])"
    Set-Location $folders[0]
    return
} 

for ($i=0; $i -lt $folders.Count; $i++) {
    Write-Host "$($i)> $($folders[$i])"
}

$selectedFolderIndex = Read-Host "Please input folder number:"
$selectedFolderPath = $folders[$selectedFolderIndex]


Info "Jump to $($selectedFolderPath)"
Set-Location $selectedFolderPath
# Write-Host "已跳轉至 $selectedFolderPath"