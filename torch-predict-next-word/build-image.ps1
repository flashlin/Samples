$command = "$env:docker_exe build -f .\Dockerfile -t char_rnn_web:dev ."
Write-Host $command
Invoke-Expression $command