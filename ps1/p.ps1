param(
    [string]$action,
    [string]$arg0
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
Import-Module "$($env:psm1HOME)/docker.psm1" -Force

function Invoke {
    param(
        [string]$cmd
    )
    $command = "$env:docker_exe $cmd"
    Write-Host $command
    Invoke-Expression $command
}

if( $action.EndsWith(".py") ) {
    InvokeCmd "python $($action)"
    return
}

if ( "build" -eq $action ) {
    $name = $arg0
    Invoke "build -f .\Dockerfile -t $($name):dev ."
    return
}

if ( "serve" -eq $action ) {
    $name = $arg0
    RemoveContainer $name
    RestartContainer $name "-p 8001:8000 $($name):dev"
    return
}

if ( "dev" -eq $action ) {
    Invoke-Expression "python ./main.py"
    return
}

if( "push" -eq $action ) {
    $ver = $arg0
    if( "" -eq $ver ) {
        Write-Host "please input push version"
        Write-Host "ex: push 1.1"
        return
    }
    Write-Host "tag image $ver"
    $container_register = "ghcr.io"
    $container_register = "docker.io"
    Invoke "tag $($name):dev $container_register/flashlin/$($name):$($ver)"
    Invoke "push $container_register/flashlin/$($name):$($ver)"
    return
}

if( "t" -eq $action ) {
    InvokeCmd "python -m unittest discover tests"
    return
}

if( "i" -eq $action ) {
    InvokeCmd "pip install -r ./requirements.txt"
    return
}

$env:PYTHONPATH="D:\VDisk\Github\Samples\py_standard"
Write-Host "run py script 1.0"
Write-Host "build: build image"
Write-Host "serve: run image"
Write-Host "dev:   run local"
Write-Host "t:     run test"
InvokeCmd "python --version"