# 初始化進度條
# $progress = 0
# $totalFolders = (Get-ChildItem -Directory -Recurse | Measure-Object).Count
# Write-Progress -Activity "尋找資料夾" -Status "進行中..." -PercentComplete 0

# # 搜尋檔案並顯示符合條件的檔案
# Get-ChildItem -Recurse | ForEach-Object {
#     if ($_.Name -like '*gguf*') {
#         $_.FullName
#     }
#     $progress++
#     $percentComplete = ($progress / $totalFolders) * 100
#     Write-Progress -Activity "尋找資料夾" -Status "進行中..." -PercentComplete $percentComplete
# }


$total = 0
function SearchDirectory {
   $folders = Get-ChildItem -Directory
   foreach ($folder in $folders) {
      #Write-Output $folder.FullName
      Write-Progress -Activity "$($folder.FullName)" -Status "$total" -PercentComplete $total
      Start-Sleep -Seconds 1
      $total += 1
   }
}

SearchDirectory

