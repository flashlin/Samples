docker exec -it queryweb /bin/bash -c 'apt-get update && apt-get install -y lsof net-tools && netstat -an | grep LISTEN | grep tcp; exec $SHELL -l'
# apt-get update & apt-get install lsof
# lsof -i -P -n | grep LISTEN
# apt-get -y install curl
# netstat -an | grep LISTEN | grep tcp
