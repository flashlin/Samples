# Use SQL Server 2022 image
FROM mcr.microsoft.com/mssql/server:2022-latest

# Create work directory
WORKDIR /usr/src/app

# Set environment variables
ENV ACCEPT_EULA=Y
ENV MSSQL_PID=Developer
ENV MSSQL_TCP_PORT=1433

# Create a config directory
RUN mkdir -p /usr/config
WORKDIR /usr/config

# Bundle config source
COPY . /usr/config

# Grant permissions for config directory
RUN chmod +x /usr/config

# Expose port 1433
EXPOSE 1433

# Set the default command
CMD ["/opt/mssql/bin/sqlservr"] 