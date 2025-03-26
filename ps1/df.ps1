param(
   [Parameter(Mandatory=$true)]
   [string]$folderName
)

Write-Host "Remove $folderName"
if ([string]::IsNullOrWhiteSpace($folderName)) {
   Write-Host "Please input folder name"
   return
}

$rootDirectory = Get-Location
$fullPath = Join-Path $rootDirectory $folderName

# 檢查資料夾是否存在
if (-not (Test-Path -Path $fullPath)) {
    Write-Host "資料夾 '$folderName' 不存在"
    return
}

Write-Host "fullPath: $fullPath"

# 刪除所有隱藏檔案（包含子資料夾中的隱藏檔案）
Get-ChildItem -Path $fullPath -File -Recurse -Force -Hidden | 
    ForEach-Object {
        Write-Host "正在刪除隱藏檔案: $($_.FullName)"
        Remove-Item -Path $_.FullName -Force
    }

# 刪除所有隱藏資料夾（包含子資料夾中的隱藏資料夾）
Get-ChildItem -Path $fullPath -Directory -Recurse -Force -Hidden | 
    ForEach-Object {
        Write-Host "正在刪除隱藏資料夾: $($_.FullName)"
        Remove-Item -Path $_.FullName -Recurse -Force
    }

Remove-Item -Path $fullPath
Write-Host "已刪除 $folderName 中的所有隱藏檔案和資料夾"
