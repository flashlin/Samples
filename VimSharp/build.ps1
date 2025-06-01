# PowerShell script to build Docker image for VimSharpApp
# 註解: Build Docker image from VimSharpApp/Dockerfile，image name 設為 vim-sharp-app

docker build -t vim-sharp-app -f ./VimSharpApp/Dockerfile ./VimSharpApp 