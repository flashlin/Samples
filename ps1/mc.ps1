param(
    [string]$action,
    [string]$args0
)

Write-Host "MiniConda"

$minicondaHome = "C:\Users\$($env:USERNAME)\AppData\Local\miniconda3\shell\condabin"
$powershellExe = "$($env:WINDIR)\System32\WindowsPowerShell\v1.0\powershell.exe"
$py_env = "C:\Users\$($env:USERNAME)\AppData\Local\miniconda3"
#& %windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command `
#    "& 'C:\Users\flash.lin71\AppData\Local\miniconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\Users\flash.lin71\AppData\Local\miniconda3'"

# %windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command 
#"& 'C:\ProgramData\Miniconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\ProgramData\Miniconda3' "

if ( "c" -eq $action ) {
    $cmd = "& $($powershellExe) -ExecutionPolicy ByPass -NoExit -Command ""& '$($minicondaHome)\conda-hook.ps1'; conda activate '$py_env'"" "
    Invoke-Expression $cmd
    return
}


if ( "c1" -eq $action ) {
    $py_env = "D:\Users\flash\miniconda3\envs\torch"
    $hookPs1 = "C:\ProgramData\Miniconda3\shell\condabin\conda-hook.ps1"
    $cmd = "& $($powershellExe) -ExecutionPolicy ByPass -NoExit -Command ""& $($hookPs1); conda activate '$py_env'"" "
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

if ( "" -eq $action ) {
    InvokeConda "env list"
    Write-Host "          : env list"
    Write-Host "c1        : use flash environment"
    Write-Host "use <name>: use name environment"
    Write-Host "n <name>  : create name environment"
    return
}

if ( "use" -eq $action ) {
    $py_env = $args0
    if ( "" -eq $py_env ) {
        Write-Host "Please input"
        Write-Host "use env-name"
        return
    }
    # D:\Users\flash\miniconda3\envs\torch
    $hookPs1 = "C:\ProgramData\Miniconda3\shell\condabin\conda-hook.ps1"
    $cmd = "& $($powershellExe) -ExecutionPolicy ByPass -NoExit -Command ""& $($hookPs1); conda activate '$py_env'"" "
    Invoke-Expression $cmd
    return
}

if( "n" -eq $action ) {
    $name = $args0
    InvokeConda "create -n ${name} python=3.10"
    return
}