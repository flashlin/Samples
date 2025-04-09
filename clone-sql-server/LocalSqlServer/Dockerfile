FROM mcr.microsoft.com/mssql/server:2022-latest

ENV ACCEPT_EULA=Y
ENV SA_PASSWORD=YourStrong@Passw0rd

# 安裝 sqlcmd 工具
USER root
RUN apt-get update \
    && apt-get install -y wget software-properties-common gnupg2 \
    && wget -qO- https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && add-apt-repository "$(wget -qO- https://packages.microsoft.com/config/ubuntu/22.04/mssql-server-2022.list)" \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18 unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/opt/mssql-tools18/bin

WORKDIR /usr/src/app

COPY CreateDatabase.sql ./
COPY init-db.sh ./
RUN chmod +x init-db.sh

USER mssql
CMD /bin/bash ./init-db.sh

EXPOSE 1433 