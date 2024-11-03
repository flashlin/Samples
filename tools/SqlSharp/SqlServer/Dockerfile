# Dockerfile
FROM mcr.microsoft.com/mssql/server:2022-latest
# FROM imunew/mssql-server-linux-with-mssql-tools
USER root

# 設定環境變數
ENV ACCEPT_EULA=Y
ENV MSSQL_SA_PASSWORD=YourStrong!Passw0rd

# 安裝 sqlcmd tool 工具
ENV DEBIAN_FRONTEND=noninteractive

# apt-get and system utilities
RUN apt-get update && apt-get install -y \
	curl apt-transport-https debconf-utils gnupg2 

# adding custom MS repository
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

# install SQL Server drivers and tools
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18
#RUN echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
#RUN /bin/bash -c "source ~/.bashrc"

RUN apt-get -y install locales \
    && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

# 將 sqlcmd 加入 PATH
ENV PATH="$PATH:/opt/mssql-tools18/bin"



# 複製初始化腳本到映像中
COPY init.sql /usr/src/app/init.sql
COPY entrypoint.sh /usr/src/app/entrypoint.sh

# 設定執行權限
RUN chmod +x /usr/src/app/entrypoint.sh

USER mssql
# 啟動 SQL Server 並執行初始化腳本
CMD ["/usr/src/app/entrypoint.sh"]