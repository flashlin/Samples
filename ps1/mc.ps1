param(
    [string]$action
)

Write-Host "MiniConda"

$minicondaHome = "C:\Users\$($env:USERNAME)\AppData\Local\miniconda3\shell\condabin"
$powershellExe = "$($env:WINDIR)\System32\WindowsPowerShell\v1.0\powershell.exe"
$py_env = "C:\Users\$($env:USERNAME)\AppData\Local\miniconda3"
#& %windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command `
#    "& 'C:\Users\flash.lin71\AppData\Local\miniconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\Users\flash.lin71\AppData\Local\miniconda3'"

if( "use" -eq $action ) {
    $cmd = "& $($powershellExe) -ExecutionPolicy ByPass -NoExit -Command ""& '$($minicondaHome)\conda-hook.ps1'; conda activate '$py_env'"" "
    Invoke-Expression $cmd
    return
}


function InvokeConda {
    param(
        [string]$parameters
    )
    $cmd = "conda " + $parameters
    Write-Host $cmd
    Invoke-Expression $cmd
}

if( "" -eq $action ) {
    InvokeConda "env list"
    return
}