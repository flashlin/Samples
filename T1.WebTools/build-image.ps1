$command = "$env:docker_exe build -f .\QueryWeb\Dockerfile -t queryweb:dev --build-arg ASPNETCORE_ENVIRONMENT=Docker ."

Write-Host $command
Invoke-Expression $command

#docker tag queryweb:dev ghcr.io/t1/queryweb:latest
#docker push ghcr.io/t1/queryweb:latest