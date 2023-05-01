FROM python:3.8-slim-buster

# 設置工作目錄
WORKDIR /app

# 將當前目錄下的所有文件複製到容器的 /app 目錄下
COPY . /app

# 安裝依賴項
RUN pip install --no-cache-dir torch flask

# 開放端口
EXPOSE 5000

# 運行命令
CMD ["python", "app.py"]