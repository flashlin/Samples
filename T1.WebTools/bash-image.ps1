docker exec -it queryweb /bin/bash -c 'apt-get update && apt-get install -y lsof && lsof -i -P -n | grep LISTEN; exec $SHELL -l'
# apt-get update & apt-get install lsof
# lsof -i -P -n | grep LISTEN
# apt-get -y install curl