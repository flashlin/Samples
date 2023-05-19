param(
    [string]$action,
    [string]$arg0
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
Import-Module "$($env:psm1HOME)/docker.psm1" -Force

$name = "predict_next_words_web"

function Invoke {
    param(
        [string]$cmd
    )
    #$command = "$env:docker_exe build -f .\Dockerfile -t predict_next_words_web:dev ."
    $command = "$env:docker_exe $cmd"
    Write-Host $command
    Invoke-Expression $command
}

if ( "build" -eq $action ) {
    Invoke "build -f .\Dockerfile -t predict_next_words_web:dev ."
    return
}

if ( "serve" -eq $action ) {
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

if( "test" -eq $action ) {
    InvokeCmd "python -m unittest discover tests"
    return
}

if( "i" -eq $action ) {
    # 注意，由于 pytorch 的原因，chromadb 目前不支持 Python 3.11
    Write-Host "install hunggingface"
    # InvokeCmd "conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.7"
    InvokeCmd "conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia"
    InvokeCmd "python install -r ./hf-req.txt"
    return
}

$env:PYTHONPATH="D:\VDisk\Github\Samples\py_standard"
Write-Host "run script 1.0"
Write-Host "build: build image"
Write-Host "serve: run image"
Write-Host "dev:   run local"