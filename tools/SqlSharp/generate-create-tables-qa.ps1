#Remove-Item \VDisk\Github\qa-pair\qa-files\synthetic\outputs\*.*

$output_path = "\VDisk\Github\qa-pair\qa-files\synthetic\outputs"


function GenerateCreateTablesQA {
    param (
        [string]$databases_folder,
        [int]$deep
    )
    $command = ".\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractCreateTableSql -i $databases_folder -o $output_path --DatabaseNamePathDeep $deep"
    Write-Host $command -ForegroundColor Green
    Invoke-Expression $command
}

# 建立資料庫
# .\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractCreateTableSql -i D:\VDisk\MyGitHub\SQL -o \VDisk\Github\qa-pair\qa-files\synthetic\outputs
# .\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractCreateTableSql -i D:\VDisk\MyGitHub\SQL -o $output_path --DatabaseNamePathDeep 6
# .\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractCreateTableSql -i D:\coredev_tw\DbProjects -o $output_path --DatabaseNamePathDeep 3
GenerateCreateTablesQA -databases_folder "D:\VDisk\MyGitHub\SQL" -deep 6
GenerateCreateTablesQA -databases_folder "D:\coredev_tw\DbProjects" -deep 3

