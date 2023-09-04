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

function AskWslNames {
    param (
    )
    $names = $result | ForEach-Object {
        $_.NAME
    }
    $name = PromptList $names
    return $name
}

if ( 's' -eq $action ) {
    $names = $result | ForEach-Object {
        $_.NAME
    }
    $name = PromptList $names
    if ( "" -eq $name ) {
        return
    }
    InvokeCmd "wsl --shutdown $name"
    return
}


if ( 'b' -eq $action ) {
    $name = AskWslNames
    if ( "" -eq $name ) {
        return
    }
    InvokeCmd "wsl -d $name"
    return
}

if ( 'rm' -eq $action ) {
    $name = AskWslNames
    if ( "" -eq $name ) {
        return
    }
    InvokeCmd "wsl --unregister $name" 
    return
}

if ( "i" -eq $action ) {
    InvokeCmd "wsl --install -d Ubuntu-22.04"
    return
}

Write-Host ""
Write-Host "i  :install Ubuntu-22.04"
Write-Host "s  :stop"
Write-Host "b  :run and enter"
Write-Host "rm :delete"
