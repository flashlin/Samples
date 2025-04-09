FROM mcr.microsoft.com/mssql/server:2022-latest

ENV ACCEPT_EULA=Y
ENV SA_PASSWORD=YourStrong@Passw0rd

WORKDIR /usr/src/app

COPY CreateDatabase.sql ./

RUN ( /opt/mssql/bin/sqlservr & ) | grep -q "Service Broker manager has started" \
    && /opt/mssql-tools/bin/sqlcmd -S localhost -U SA -P YourStrong@Passw0rd -i CreateDatabase.sql

EXPOSE 1433

CMD [ "/opt/mssql/bin/sqlservr" ] 