# 假設 $input_list 和 $exclude_list 是這樣的
$input_list = @("apple\", "banana\", "1banana\", "date", "elderberry", "C:\Users\flash\OneDrive\mssql-lab")
$exclude_list = @("banana\", "date")

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

# 輸出結果
# $file_list


function filter_list {
    param (
        [Parameter(Mandatory=$true)]
        [string[]]$input_list,  # 輸入的字串陣列
        [Parameter(Mandatory=$true)]
        [string]$searchTerm  # 輸入的字串陣列
    )
    $result = $input_list | Where-Object { $_ -like "*$searchTerm*" }
    return $result
}
$file_list = filter_list -input_list $input_list -searchTerm "flash"
$file_list