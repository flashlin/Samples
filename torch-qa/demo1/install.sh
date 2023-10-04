#!/bin/bash
set -e
sudo apt install mysql-client -y
which mysql
mysql --version

echo "docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 107794038a3b"
# mysql -h 172.18.0.2 -P 3306 --protocol=tcp -u flash -p