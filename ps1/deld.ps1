param(
    [string]$folder
)

if( "" -eq $folder ) {
    Write-Host "Fast Delete Folder Script"
    Write-Host "deld <folder>"
    return
}

Write-Host "Deleting $folder files..."
Remove-Item -Path "$($folder)*" -Recurse #-Force

