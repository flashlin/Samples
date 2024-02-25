param(
   [string]$folderName
)
Write-Host "Remove $folderName"
if ("" -eq $folderName) {
   Write-Host "Please input folder name"
   return
}
$rootDirectory = Get-Location
Get-ChildItem -Path $rootDirectory -Directory -Recurse | Where-Object { $_.Name -eq $folderName } | ForEach-Object {
   Write-Host "Removing $($_.FullName)"
   Remove-Item -Path $_.FullName -Recurse -Force
}
