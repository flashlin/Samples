$search_list = @($args)

function exclude_file_list {
    param (
        [Parameter(Mandatory = $true)]
        [string[]]$input_list
    )
    $exclude_list = @(
        "Program Files\", "Program Files (x86)\", "ProgramData\", 
        "Windows\", "AppData\Local\", "RECYCLE.BIN\",
        ".Continue\",
        ".git\", ".idea\", "node_modules\", ".deploy_git\",
        ".nuget\", ".vscode\", ".vs\")
    # 過濾 $input_list 中不包含任何 $exclude_list 中元素的項目
    $file_list = $input_list | Where-Object {
        $item = $_
        $shouldExclude = $false
        foreach($exclude in $exclude_list) {
            if($item -like "*$exclude*") {
                $shouldExclude = $true
                break
            }
        }
        -not $shouldExclude
    }
    return $file_list
}

function search_folder {
    param (
        [Parameter(Mandatory = $true)]
        [string]$searchTerm
    )
    # 建立命令字串
    $command = "es.exe /ad -s `"$searchTerm`""
    Write-Host $command
    $file_list = Invoke-Expression $command
    if( $null -eq $file_list ) {
        return @()
    }
    $file_list = exclude_file_list -input_list $file_list
    # Write-Host "search found" $file_list.Count
    return $file_list
}


function show_menu {
    param (
        [Parameter(Mandatory=$true)]
        [string[]]$input_list  # 輸入的字串陣列
    )

    # 使用 fzf 進行選擇
    $selected = $input_list | fzf --color=dark --height=30%

    # 返回選擇的結果
    return $selected
}

function show_list {
    param (
        [Parameter(Mandatory=$true)]
        [string[]]$file_list  # 輸入的字串陣列
    )
    $file_list | ForEach-Object {
        Write-Host $_
    }
    Write-Host "----------"
    Write-Host ""
    Write-Host ""
    Write-Host ""
}

function filter_list {
    param (
        [Parameter(Mandatory=$true)]
        [string[]]$input_list,  # 輸入的字串陣列
        [Parameter(Mandatory=$true)]
        [string]$searchTerm  # 輸入的字串陣列
    )
    # show_list -file_list $input_list
    $result = $input_list | Where-Object { $_ -like "*$searchTerm*" }
    # show_list -file_list $result
    return $result
}

function Search-FilterItems {
    param(
        [Parameter(Mandatory)]
        [string[]]$SearchList
    )

    $result = search_folder -searchTerm $SearchList[0]
    #Write-Host "First: " $result $result.Count
    
    if ($result.Count -eq 1) {
        return $result
    }
    
    $index = 1
    while ($result.Count -gt 1 -and $index -lt $SearchList.Count) {
        #Write-Host "Search: " $SearchList[$index]
        $result = filter_list -input_list $result -searchTerm $SearchList[$index]
        #Write-Host "Search: " $result $result.Count
        if ($result.Count -eq 1) {
            return $result[0]
        }
        $index++
    }

    return $result
}

Write-Host "Fast jump to location: $search_list"

$count = 0
$selected = $null
$file_list = @()

$result = Search-FilterItems -SearchList $search_list
if( $result.Count -gt 1  ) {
    $selected = show_menu -input_list $result
}  
if( $result.Count -eq 1  ) {
    $selected = $result
}  

# $search_list | ForEach-Object { 
#     $count++
#     $searchTerm = $_
#     if (-not $file_list) {
#         # 呼叫搜尋
#         Write-Host "搜尋 $searchTerm"
#         $file_list = search_folder -searchTerm $searchTerm
#         # Write-Host $file_list.GetType()
#         if( $file_list -eq $null ) {
#             # Write-Host "找不到"
#             $file_list = @()
#             $selected = $null
#             return
#         }
#         if( "string" -eq "$($file_list.GetType())" ) {
#             # Write-Host "只有一個"
#             $selected = $file_list
#             return
#         }
#         if( $file_list.Count -eq 0 ) {
#             return
#         }
#         # 找到很多個 
#         Write-Host "找到 $($file_list.Count) 個"
#     }

#     if ( $count -eq $search_list.Count ) {
#         # 最後一個
#         # Write-Host $file_list
#         # Write-Host "最後過濾 $searchTerm"
#         $result = filter_list -input_list $file_list -searchTerm $searchTerm
#         # Write-Host "最後過濾結果 $($result.Count)"
#         if( $result.Count -gt 1  ) {
#             $selected = show_menu -input_list $result
#         }  
#         if( $result.Count -eq 1  ) {
#             $selected = $result[0]
#         }  
#     } else {
#         # 還有更多的條件
#         Write-Host "呼叫過濾 $searchTerm"
#         $result = filter_list -input_list $file_list -searchTerm $searchTerm
#         if( $result.Count -ne 0  ) {
#             $file_list = $result
#         } 
#     }
# }

# 輸出選擇的結果
if ($null -ne $selected) {
    Write-Host "Selected file: $selected"
    Set-Location $selected
} else {
    Write-Host "Not found any folder."
}