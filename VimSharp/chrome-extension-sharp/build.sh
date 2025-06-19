docker build -t support-tool .
docker rm -f support-tool
docker run -d -p 8080:80 --name support-tool support-tool