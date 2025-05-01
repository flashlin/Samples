# 創建一個空的 ArrayList 來存儲結果
$PathCommitedList = [System.Collections.ArrayList]::new()

# 初始化變數
$processedPaths = @{}
$maxPaths = 30
$skip = 0
$batchSize = 50

# 持續搜索直到找到足夠的不同路徑
while ($processedPaths.Count -lt $maxPaths) {
    # 獲取一批 git log 記錄
    $gitLog = git log --name-status --pretty=format:"%cd" --date=format:"%Y/%m/%d %H:%M:%S" --skip=$skip -n $batchSize

    # 如果沒有更多記錄，則退出循環
    if (-not $gitLog) {
        break
    }

    $currentDate = $null
    
    # 處理每一行輸出
    $gitLog | ForEach-Object {
        # 如果已經找到足夠的路徑，則跳出
        if ($processedPaths.Count -ge $maxPaths) {
            return
        }

        $line = $_
        
        # 如果行包含日期（不以字母開頭），則更新當前日期
        if ($line -match "^\d{4}/\d{2}/\d{2}") {
            $currentDate = $line
        }
        # 如果行以 A/M/D（新增/修改/刪除）開頭，則處理文件路徑
        elseif ($line -match "^[AMD]\s+(.+)$") {
            $filePath = $matches[1]
            $pathParts = $filePath -split '/'
            
            # 跳過只有檔名的情況
            if ($pathParts.Count -lt 2) {
                return
            }
            
            # 獲取兩層路徑（不包含檔名）
            $twoLevelPath = if ($pathParts.Count -gt 2) {
                "$($pathParts[0])/$($pathParts[1])"
            } else {
                "$($pathParts[0])"  # 如果只有一層目錄加檔名，則只取目錄
            }
            
            # 如果這個路徑還沒有被處理過且不為空
            if (-not [string]::IsNullOrEmpty($twoLevelPath) -and -not $processedPaths.ContainsKey($twoLevelPath)) {
                $processedPaths[$twoLevelPath] = $true
                
                # 創建新的對象並添加到列表中
                $item = [PSCustomObject]@{
                    Path = $twoLevelPath
                    Date = [DateTime]::ParseExact($currentDate, "yyyy/MM/dd HH:mm:ss", $null)
                }
                [void]$PathCommitedList.Add($item)
                
                Write-Host "找到新路徑: $twoLevelPath - $currentDate"
            }
        }
    }
    
    $skip += $batchSize
}

Write-Host "`n總共找到 $($processedPaths.Count) 個不同的路徑`n"

# 按日期排序（從最近到最遠）並顯示結果
$PathCommitedList | Sort-Object Date -Descending | Format-Table @{
    Label = "Path"
    Expression = { $_.Path }
    Width = 40  # 增加寬度以適應更長的路徑
}, @{
    Label = "Date"
    Expression = { $_.Date.ToString("yyyy/MM/dd HH:mm:ss") }
    Width = 20
}
