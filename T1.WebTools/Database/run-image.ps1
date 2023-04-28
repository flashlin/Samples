$docker_exe = $env:docker_exe
Write-Host "docker start query-db"
# $rc = & docker ps --filter name=query-db
# docker start queru-db
# docker run -it --name query-db -p 4331:1433 query-db
if (Invoke-Expression "$docker_exe ps -q --filter name=query-db") {
   Invoke-Expression "$docker_exe ps --filter name=query-db"
   Write-Host "The container is already started"
   return
}

if (Invoke-Expression "$docker_exe ps -aq --filter name=query-db") {
   Write-Output "Container exists and is stopped, restart it"
   Invoke-Expression "$docker_exe start query-db"
   exit
}


Write-Output "start"
Invoke-Expression "$docker_exe run -it --name query-db -p 4331:1433 query-db"