param(
    [string]$action
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

$result = InvokeCmd "wsl -l -v" | SplitTableString
Write-Host $result
