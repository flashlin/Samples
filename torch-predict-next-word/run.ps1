param(
    [string]$action
)

function Invoke{
    param(
        [string]$cmd
    )
    #$command = "$env:docker_exe build -f .\Dockerfile -t predict_next_words_web:dev ."
    $command = "$env:docker_exe $cmd"
    Write-Host $command
    Invoke-Expression $command
}

if( "build" -eq $action ) {
    Invoke "build -f .\Dockerfile -t predict_next_words_web:dev ."
    return
}

if( "serve" -eq $action ) {
    Invoke "run -it --name predict_next_words_web -p 5001:8000 predict_next_words_web:dev"
    return
}

Write-Host "run script 1.0"
Write-Host "build: build image"
Write-Host "serve: run image"