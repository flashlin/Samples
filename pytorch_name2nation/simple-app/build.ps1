docker build -t simpleapp .
# docker run -d --name myapp -p 80:80 simpleapp
docker run -it --name myapp -p 8081:80 simpleapp
# http://localhost:8081/docs
