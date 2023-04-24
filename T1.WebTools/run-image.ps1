Invoke-Expression "$($env:docker_exe) rm queryweb"
Invoke-Expression "$($env:docker_exe) run -it -p 5001:8000 --name queryweb docker.io/flashlin/queryweb:1.0"