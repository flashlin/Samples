param(
    [string]$folder
)
$output = & es.exe "folder:$folder"
$folders = $output -split "`r`n"

if( $folders -eq 1 ) {
    Set-Location $$folders[0]
    return
} 

for ($i=0; $i -lt $folders.Count; $i++) {
    Write-Host "$($i)> $($folders[$i])"
}

$selectedFolderIndex = Read-Host "Please input folder number:"
$selectedFolderPath = $folders[$selectedFolderIndex]

Set-Location $selectedFolderPath
# Write-Host "已跳轉至 $selectedFolderPath"