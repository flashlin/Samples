$docker_exe = $env:docker_exe
Invoke-Expression "$docker_exe build --no-cache -t query-db ."