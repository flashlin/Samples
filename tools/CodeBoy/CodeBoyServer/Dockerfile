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
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS runtime
WORKDIR /app

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy the published application
COPY --from=build /app/publish .

# Change ownership to appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port that ASP.NET Core listens on
EXPOSE 8080

# Set environment variables
ENV ASPNETCORE_URLS=http://+:8080
ENV ASPNETCORE_ENVIRONMENT=Production

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
ENTRYPOINT ["dotnet", "CodeBoyServer.dll"]
