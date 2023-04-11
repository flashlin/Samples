Invoke-Expression "$env:docker_exe rm queryweb"
Invoke-Expression "$env:docker_exe run -it -p 5001:80 --name queryweb queryweb:dev"