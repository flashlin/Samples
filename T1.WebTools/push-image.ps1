Write-Host "https://github.com/settings/tokens"
# $MY_TOKEN = Get-Content -Path "d:/VDisk/Devp/github-packages.key"
# $env:CR_PAT = $MY_TOKEN
# $env:CR_PAT | docker login ghcr.io -u flash.lin@gmail.com --password-stdin

$ver = "1.3"

Write-Host "tag image $ver"
$container_register = "ghcr.io"
$container_register = "docker.io"
Invoke-Expression "$env:docker_exe tag queryweb:dev $container_register/flashlin/queryweb:$($ver)"
Invoke-Expression "$env:docker_exe push $container_register/flashlin/queryweb:$($ver)"
