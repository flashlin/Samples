param(
    [Parameter(Mandatory=$true)]
    [string]$filePattern
)

function DeleteFolderAllFile {
    param(
        [Parameter(Mandatory=$true)]
        [string]$targetFolder,
        [Parameter(Mandatory=$true)]
        [string]$pattern
    )
    Write-Host "Deleting $targetFolder '$pattern' files..."
    $files = Get-ChildItem -Path $targetFolder -Recurse -Filter $pattern
    $totalFiles = $files.Count
    $progress = 0
    foreach ($file in $files) {
        $progress++
        $percentComplete = ($progress / $totalFiles) * 100
        $status = "Delete progress: {0}% ({1}/{2})" -f $percentComplete, $progress, $totalFiles
        Write-Progress -Activity "Deleting file" -Status $status -PercentComplete $percentComplete
        Remove-Item $file.FullName -Force
    }
    Write-Progress -Activity "Delete Complete" -Status "Delete progress: 100% ($totalFiles/$totalFiles)" -Completed
    Write-Host "Delete $totalFiles File Complete."
}

#$filePattern = "*.webp"
if( "" -ne $filePattern ) {
    # Get-ChildItem -Path ./ -Recurse -Filter $filePattern | Remove-Item -Force -Recurse
    DeleteFolderAllFile "./" -pattern $filePattern
    return
}

Write-Host "f-del *.webp    刪除所有 .webp 包含子資料夾都刪除"
