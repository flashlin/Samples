Write-Host "docker start query-db"
# $rc = & docker ps --filter name=query-db
# docker start queru-db
# docker run -it --name query-db -p 4331:1433 query-db
if (docker ps -q --filter name=query-db) {
   docker ps --filter name=query-db
   Write-Host "The container is already started"
   return
}

if (docker ps -aq --filter status=exited --filter name=query-db) {
   Write-Output "Container exists and is stopped, restart it"
   docker start query-db
   exit
}


Write-Output "start"
docker run -it --name query-db -p 4331:1433 query-db