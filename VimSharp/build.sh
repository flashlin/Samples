# PowerShell script to build Docker image for VimSharpApp
# 註解: Build Docker image from VimSharpApp/Dockerfile，image name 設為 vim-sharp-app

docker build -t vim-sharp-app -f ./Dockerfile .
docker rm -f vim-sharp-app
docker run -d -p 8081:8080 vim-sharp-app --name vim-sharp-app