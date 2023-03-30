docker stop queru-db
docker rm query-db
docker run -it --name query-db -p 4331:1433 query-db