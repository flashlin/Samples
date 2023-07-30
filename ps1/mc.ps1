param(
    [string]$action,
    [string]$args0,
    [string]$args1
)
Import-Module $env:psm1Home\common.psm1 -Force

Write-Host "MiniConda"

$minicondaHome = "C:\Users\$($env:USERNAME)\AppData\Local\miniconda3\shell\condabin"
$powershellExe = "$($env:WINDIR)\System32\WindowsPowerShell\v1.0\powershell.exe"
$py_env = "C:\Users\$($env:USERNAME)\AppData\Local\miniconda3"
#& %windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command `
#    "& 'C:\Users\flash.lin71\AppData\Local\miniconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\Users\flash.lin71\AppData\Local\miniconda3'"

# %windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command 
#"& 'C:\ProgramData\Miniconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\ProgramData\Miniconda3' "

if ( "c1" -eq $action ) {
    $cmd = "& $($powershellExe) -ExecutionPolicy ByPass -NoExit -Command ""& '$($minicondaHome)\conda-hook.ps1'; conda activate '$py_env'"" "
    Invoke-Expression $cmd
    return
}


if ( "c0" -eq $action ) {
    $minicondaHome = "C:\Users\$($env:USERNAME)\miniconda3"

    $hookPs1 = "C:\ProgramData\Miniconda3\shell\condabin\conda-hook.ps1"
    $cmd = "& $($powershellExe) -ExecutionPolicy ByPass -NoExit -Command ""& '$($minicondaHome)\shell\condabin\conda-hook.ps1' ; conda activate '$($minicondaHome)' """
    # $cmd = "& $($powershellExe) -ExecutionPolicy ByPass -NoExit -Command ""& $($hookPs1); conda activate '$py_env'"" "
    Invoke-Expression $cmd
    return
}


function InvokeConda {
    param(
        [string]$parameters
    )
    $cmd = "conda " + $parameters
    Info $cmd
    Invoke-Expression $cmd
}

if ( "" -eq $action ) {
    InvokeConda "env list"
    Write-Host "                      : env list"
    Write-Host "c0 / c1               : use flash environment"
    Write-Host "t                     : use torch environment"
    Write-Host "use <name>            : use name environment"
    Write-Host "n <name> [python-ver] : create name environment 3.9"
    Write-Host "c <name>              : switch to name environment"
    Write-Host "rm <name>             : remove name environment"
    Write-Host "i                     : install conda packages"
    Write-Host "clean                 : clean unused conda packages"
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

if ( "n" -eq $action ) {
    $name = $args0
    $pythonVer = $args1
    if ( "" -eq $pythonVer ) {
        $pythonVer = "3.10"
    }
    InvokeConda "create -n ${name} python=${pythonVer}"
    return
}

if ( "c" -eq $action ) {
    $name = $args0
    InvokeConda "activate $name"
    return
}

if( "update" -eq $action){
    InvokeCmd "conda update -n base -c defaults conda"
    return
}

if ( "i" -eq $action ) {
    # for linux
    # conda install -c conda-forge bitsandbytes
    #InvokeCmd "conda update -n base -c defaults conda"
    InvokeConda "install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia"
    #InvokeConda "install -c conda-forge faiss"
    InvokeConda "install -c conda-forge faiss-gpu"
    InvokeCmd "pip install pandas tensorboard"
    return
}

if ( "rm" -eq $action ) {
    $name = $args0
    InvokeConda "env remove -n $name"
    return
}

if ( "t" -eq $action ) {
    InvokeConda "activate torch"
    return
}

if ( "clean" -eq $action ) {
    InvokeConda "clean --packages"
    return
}
