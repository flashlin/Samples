#!/bin/bash
set -e
docker-compose up -d
# docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mysql_gpt_db
# mysql -h `docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mysql_gpt_db` -P 3306 --protocol=tcp -u flash -p
# mysql -h 127.0.0.1 -P 3306 -u flash -p


echo "localhost: 6333 for Qdrant service"

docker ps
