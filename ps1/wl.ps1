param(
    [string]$action
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

Write-Host "wsl script"

$result = InvokeCmd "wsl -l -v" | SplitTableString
$result | ForEach-Object {
    Write-Host $_
}

if( 's' -eq $action ) {
    $names = $result | ForEach-Object {
        $_.NAME
    }
    $name = PromptList $names
    if( "" -eq $name ) {
        return
    }
    InvokeCmd "wsl --shutdown $name"
    return
}

Write-Host ""
Write-Host "s: stop"
