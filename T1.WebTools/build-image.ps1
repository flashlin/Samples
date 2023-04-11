docker build -f .\QueryWeb\Dockerfile -t queryweb:dev --build-arg ASPNETCORE_ENVIRONMENT=docker .
#docker tag queryweb:dev ghcr.io/t1/queryweb:latest
#docker push ghcr.io/t1/queryweb:latest