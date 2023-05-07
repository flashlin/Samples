param(
    [string]$action
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force

if( "" -eq $action ) {
    Write-Host "test"
    return
}

if( "test" -eq $action ) {
    InvokeCmd "python -m unittest discover tests"
    return
}