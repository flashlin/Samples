docker rm queryweb
docker run -it -p 5001:80 --name queryweb queryweb:dev