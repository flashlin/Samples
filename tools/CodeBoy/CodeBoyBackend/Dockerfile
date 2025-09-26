# CodeBoy Server Dockerfile with multi-stage build
# Stage 1: Build
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy project files for restore
COPY CodeBoyLib/CodeBoyLib.csproj CodeBoyLib/
COPY CodeBoyServer/CodeBoyServer.csproj CodeBoyServer/

# Restore dependencies
RUN dotnet restore CodeBoyServer/CodeBoyServer.csproj

# Copy source code
COPY CodeBoyLib/ CodeBoyLib/
COPY CodeBoyServer/ CodeBoyServer/

# Build the application
WORKDIR /src/CodeBoyServer
RUN dotnet build -c Release --no-restore

# Publish the application
RUN dotnet publish -c Release --no-build -o /app/publish

# Stage 2: Runtime
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS runtime
WORKDIR /app

# Install curl for health checks and other utilities
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy the published application
COPY --from=build /app/publish .

# Ensure appuser has access to temp directories for dynamic compilation and .NET operations
RUN mkdir -p /tmp/.nuget /tmp/.dotnet && \
    chmod 777 /tmp && \
    chown -R appuser:appuser /tmp

# Change ownership to appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port that ASP.NET Core listens on
EXPOSE 8080

# Set environment variables
ENV HTTP_PORTS=8080
ENV ASPNETCORE_ENVIRONMENT=Production

# Configure .NET environment variables to use /tmp for all dotnet operations
ENV DOTNET_CLI_HOME=/tmp
ENV DOTNET_USER_PROFILE=/tmp
ENV HOME=/tmp
ENV NUGET_PACKAGES=/tmp/.nuget
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
ENTRYPOINT ["dotnet", "CodeBoyServer.dll"]
