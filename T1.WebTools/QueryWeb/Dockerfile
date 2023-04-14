FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["QueryWeb/QueryWeb.csproj", "QueryWeb/"]
COPY ["QueryKits/QueryKits.csproj", "QueryKits/"]
COPY ["QueryRazor/QueryRazor.csproj", "QueryRazor/"]
COPY ["T1.WebTools/T1.WebTools.csproj", "T1.WebTools/"]

RUN dotnet restore "QueryWeb/QueryWeb.csproj"
COPY . .
WORKDIR "/src/QueryWeb"
RUN dotnet build "QueryWeb.csproj" -c Release -o /app/build /p:ASPNETCORE_ENVIRONMENT=Docker


FROM build AS publish
RUN dotnet publish "QueryWeb.csproj" -c Release -o /app/publish /p:UseAppHost=false

#
COPY --from=build /src/Database /app/publish/Database 
COPY --from=build /src/QueryWeb/entrypoint.sh /app/publish
RUN chmod +x /app/publish/entrypoint.sh


FROM base AS final
LABEL org.opencontainers.image.source=https://github.com/octocat/my-repo
LABEL org.opencontainers.image.description="My container image"
LABEL org.opencontainers.image.licenses=MIT

# Install sql server
USER root
RUN apt-get update && \
    apt-get install -y curl gnupg2 apt-transport-https wget software-properties-common && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    add-apt-repository "$(wget -qO- https://packages.microsoft.com/config/ubuntu/20.04/mssql-server-2022.list)" && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools unixodbc-dev mssql-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 


WORKDIR /app
COPY --from=publish /app/publish .
ENV DOTNET_ENVIRONMENT=Docker
# ENTRYPOINT ["dotnet", "QueryWeb.dll"]
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
# ENTRYPOINT ["/app/entrypoint.sh"]
# ENTRYPOINT ["/bin/bash"]