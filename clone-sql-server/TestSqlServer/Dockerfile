# Use SQL Server 2022 image
FROM mcr.microsoft.com/mssql/server:2022-latest

# Switch to root to create directories
USER root

# Create work directory
WORKDIR /usr/src/app

# Set environment variables
ENV ACCEPT_EULA=Y
ENV MSSQL_PID=Developer
ENV MSSQL_TCP_PORT=1433
ENV MSSQL_SA_PASSWORD=YourStrongPassw0rd!

# Create a config directory
RUN mkdir -p /usr/config
WORKDIR /usr/config

# Copy initialization script
COPY init.sql /usr/config/
COPY ./setup.sh /usr/config/

# Grant permissions
RUN chmod +x /usr/config/setup.sh && \
    chown -R mssql:root /usr/config

# Switch back to mssql user
USER mssql

# Expose port 1433
EXPOSE 1433

# Start SQL Server, run the initialization script, and then loop forever
CMD /bin/bash ./setup.sh 