Write-Host "docker start query-db"
$rc = & docker ps --filter name=query-db
# docker start queru-db
# docker run -it --name query-db -p 4331:1433 query-db