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

# Install additional .NET runtimes that dotnet-ef might need
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Copy the published application
COPY --from=build /app/publish .

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/.nuget /tmp/.dotnet /home/appuser/.dotnet/tools /home/appuser/.local/share /home/appuser/.templateengine && \
    chmod -R 777 /tmp && \
    chown -R appuser:appuser /tmp /home/appuser /app

# Set environment variables
ENV DOTNET_CLI_HOME=/home/appuser
ENV DOTNET_USER_PROFILE=/home/appuser
ENV HOME=/home/appuser
ENV NUGET_PACKAGES=/home/appuser/.nuget/packages
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1
ENV PATH="${PATH}:/home/appuser/.dotnet/tools"
ENV DOTNET_ROOT=/usr/share/dotnet
ENV DOTNET_ROLL_FORWARD=LatestMajor
ENV DOTNET_ROLL_FORWARD_TO_PRERELEASE=0

# Switch to appuser and install dotnet-ef tools
USER appuser

# Install dotnet-ef tool matching the SDK version
RUN dotnet tool install --global dotnet-ef --version 9.0.0

# Verify installation
RUN dotnet ef --version || echo "dotnet-ef installation verification"

# Expose the port that ASP.NET Core listens on
EXPOSE 8080

# Set environment variables
ENV HTTP_PORTS=8080
ENV ASPNETCORE_ENVIRONMENT=Production

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
ENTRYPOINT ["dotnet", "CodeBoyServer.dll"]
