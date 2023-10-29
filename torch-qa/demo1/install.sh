#!/bin/bash
set -e
sudo apt install mysql-client -y
which mysql
mysql --version
echo "docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 107794038a3b"
# mysql -h 172.18.0.2 -P 3306 --protocol=tcp -u flash -p



# https://spacy.io/usage
echo install spacy
conda install -c conda-forge spacy
python -m spacy download en_core_web_trf
python -m spacy download xx_sent_ud_sm