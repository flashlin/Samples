#!/bin/bash
set -e
docker-compose up -d
# docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mysql_gpt_db
mysql -h `docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mysql_gpt_db` -P 3306 --protocol=tcp -u flash -p