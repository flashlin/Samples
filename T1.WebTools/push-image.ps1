Write-Host "https://github.com/settings/tokens"
$MY_TOKEN = Get-Content -Path "d:/VDisk/Devp/github-packages.key"
$env:CR_PAT = $MY_TOKEN
$env:CR_PAT | docker login ghcr.io -u flash.lin@gmail.com --password-stdin

Write-Host "tag image"
Invoke-Expression "$env:docker_exe tag queryweb:dev ghcr.io/flashlin/queryweb:latest"
Invoke-Expression "$env:docker_exe push ghcr.io/flashlin/queryweb:latest"
