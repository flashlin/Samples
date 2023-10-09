#!/bin/bash
set -e
docker-compose down
docker rm mysql_gpt_db -f
docker rmi demo1-mysql -f
./start.sh
