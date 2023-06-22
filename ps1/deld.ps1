param(
    [string]$folder
)

if( "" -eq $folder ) {
    Write-Host "Fast Delete Folder Script"
    Write-Host "deld <folder>"
    return
}

Remove-Item -Path "$($folder)*" -Recurse #-Force

