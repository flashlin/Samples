if [ "$(docker ps -q --filter name=query-db)" ]; then
  echo "the query-db is already started"
  exit
fi

if [ ! -z "$(docker ps -aq --filter status=exited --filter name=query-db)" ]; then
  echo "Container exists and is stopped, restart it"
  docker start query-db
  exit
fi

echo "start"
docker run -it --name query-db -p 4331:1433 query-db