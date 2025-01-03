#Remove-Item \VDisk\Github\qa-pair\qa-files\synthetic\outputs\*.*

$output_path = "\VDisk\Github\qa-pair\qa-files\synthetic\outputs"

function GenerateSelectQA {
    param (
        [string]$databases_folder,
        [int]$deep
    )
    $command = ".\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractSelectSql -i $databases_folder -o $output_path --DatabaseNamePathDeep $deep"
    Write-Host $command -ForegroundColor Green
    Invoke-Expression $command
}

GenerateSelectQA -databases_folder "D:\VDisk\MyGitHub\SQL" -deep 6
GenerateSelectQA -databases_folder "D:\coredev_tw\DbProjects" -deep 3
