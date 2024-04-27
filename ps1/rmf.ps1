param(
   [string]$folderName
)


if( "" -eq $folderName ) {
   Write-Host "Fast Delete Folder Script"
   Write-Host "rmf <folder>"
   return
}

# 搜尋目錄函式
function Search-And-DeleteBin {
   param (
       [string]$rootDirectory
   )

   # 搜尋目錄
   Get-ChildItem -Path $rootDirectory -Recurse -Directory | ForEach-Object {
      $subDirectory = $_.FullName
      # 檢查子目錄是否名稱為 "bin"
      if ($_.Name -eq $folderName) {
         # 刪除 bin 子目錄
         Remove-Item -Path $subDirectory -Recurse -Force
         Write-Host "已刪除目錄: $subDirectory" -ForegroundColor Yellow
      } else {
         # 遞迴進入子目錄搜尋
         Search-And-DeleteBin -rootDirectory $subDirectory
      }
   }
}

# 呼叫函式，指定要搜尋的根目錄
Search-And-DeleteBin -rootDirectory .