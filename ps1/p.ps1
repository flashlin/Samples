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

if( "hf" -eq $action ) {
    $apiKey = ReadFile "d:/demo/huggingface-api-key.txt"
    $env:HUGGING_FACE_HUB_TOKEN = $apiKey
    Write-Host "set HUGGING_FACE_HUB_TOKEN='${apiKey}'"
    return
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

if( "logs" -eq $action ) {
    InvokeCmd "tensorboard --logdir=./logs"
    return
}

if( "info" -eq $action ) {
    InvokeCmd "nvidia-smi"
    $code = "import torch`nprint(torch.cuda.is_available())"
    InvokeCmd "python -c '$code'"
    # pip install bitsandbytes-cuda117
    # pip install https://github.com/acpopescu/bitsandbytes/releases/download/v0.37.2-win.1/bitsandbytes-0.37.2-py3-none-any.whl
    # C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin\nvcc.exe --version
    # D:\Users\flash\miniconda3\pkgs\cuda-demo-suite-11.8.86-0\demo_suite\bandwidthTest.exe
    return
}

$env:PYTHONPATH="D:\VDisk\Github\Samples\py_standard"
Write-Host "run py script 1.0"
Write-Host "build: build image"
Write-Host "serve: run image"
Write-Host "dev:   run local"
Write-Host "t:     run test"
Write-Host "hf:    set huggingface token"
InvokeCmd "python --version"