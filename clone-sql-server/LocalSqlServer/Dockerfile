# Build stage for database initialization
FROM mcr.microsoft.com/mssql/server:2022-latest AS builder

ARG SA_PASSWORD
ENV ACCEPT_EULA=Y
ENV SA_PASSWORD=${SA_PASSWORD}
ENV MSSQL_ENCRYPT=OPTIONAL
ENV SQLCMDPASSWORD=${SA_PASSWORD}

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

# Create logs directory and set permissions
RUN mkdir -p /var/log/sqlserver \
    && chown -R mssql:mssql /var/log/sqlserver \
    && mkdir -p /var/opt/mssql-initialized \
    && chown -R mssql:mssql /var/opt/mssql-initialized

# Copy initialization files
COPY CreateDatabase.sql ./
COPY create-db.sh ./
RUN chmod +x create-db.sh \
    && chown -R mssql:mssql .

# Initialize database
USER mssql
RUN /bin/bash ./create-db.sh \
    && cp -R /var/opt/mssql/* /var/opt/mssql-initialized/

# Final stage
FROM mcr.microsoft.com/mssql/server:2022-latest

ARG SA_PASSWORD
ENV ACCEPT_EULA=Y
ENV SA_PASSWORD=${SA_PASSWORD}
ENV MSSQL_ENCRYPT=OPTIONAL

# Copy initialized database files
COPY --from=builder /var/opt/mssql-initialized /var/opt/mssql
COPY --from=builder /var/log/sqlserver /var/log/sqlserver

USER mssql
CMD ["/opt/mssql/bin/sqlservr"]

EXPOSE 1433 